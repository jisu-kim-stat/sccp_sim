#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


API_BASE = "https://api.inaturalist.org/v1"


def load_categories_from_train_json(train_json_path: str) -> List[Dict[str, Any]]:
    """
    train2017.json / val2017.json 에는 보통
    dict_keys(['info','images','licenses','annotations','categories'])
    형태로 들어있고 categories 원소는 {id, name, supercategory}.
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = data["categories"]
    # 안정적으로 key 존재 확인
    for c in cats:
        if not all(k in c for k in ("id", "name", "supercategory")):
            raise ValueError(f"category missing keys: {c.keys()}")
    return cats


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def req_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "inat-taxonomy-meta/0.1"})
    r.raise_for_status()
    return r.json()


def get_taxon_by_id(taxon_id: int, include_ancestors: bool = True) -> Optional[Dict[str, Any]]:
    """
    GET /v1/taxa/{id}
    - 응답 JSON에 results[0]가 taxon record.
    - ancestors 필드가 항상 오진 않을 수 있어 include 파라미터도 같이 시도.
    """
    url = f"{API_BASE}/taxa/{taxon_id}"
    params = {}
    # iNat API는 include 파라미터로 연관 객체를 주는 경우가 많아서 시도
    if include_ancestors:
        params["include"] = "ancestors"
    try:
        js = req_json(url, params=params)
        if "results" not in js or len(js["results"]) == 0:
            return None
        return js["results"][0]
    except Exception:
        return None


def search_taxon_by_name(name: str, rank: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    fallback: GET /v1/taxa?q=NAME
    - name이 정확 매칭이 아닐 수 있어서 첫 결과를 쓰되,
      가능한 경우 scientific_name == name 같은 조건을 우선 시도.
    """
    url = f"{API_BASE}/taxa"
    params = {"q": name, "per_page": 30}
    if rank is not None:
        params["rank"] = rank
    try:
        js = req_json(url, params=params)
        res = js.get("results", [])
        if not res:
            return None

        # 1) 최대한 name과 일치하는 걸 우선 선택
        # iNat taxon record에는 name(학명) / preferred_common_name 등 다양하게 있음
        name_l = name.strip().lower()
        exact = None
        for t in res:
            sci = str(t.get("name", "")).strip().lower()
            if sci == name_l:
                exact = t
                break
        return exact if exact is not None else res[0]
    except Exception:
        return None


def find_family_from_taxon(taxon: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """
    taxon record에서 family taxon을 찾아 (family_id, family_name) 반환.
    - ancestors 안에 rank == 'family' 가 있으면 그걸 사용
    - 없으면 자기 자신이 family인지 확인
    """
    # 1) ancestors에서 찾기
    anc = taxon.get("ancestors", None)
    if isinstance(anc, list):
        for a in anc:
            if str(a.get("rank", "")).lower() == "family":
                return a.get("id", None), a.get("name", None)

    # 2) 자기 자신이 family인 경우
    if str(taxon.get("rank", "")).lower() == "family":
        return taxon.get("id", None), taxon.get("name", None)

    return None, None


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, required=True, help="path to train2017.json (or val2017.json)")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument("--sleep", type=float, default=0.08, help="sleep seconds between API calls (polite)")
    ap.add_argument("--max_retry", type=int, default=5)
    args = ap.parse_args()

    train_json = os.path.expanduser(args.train_json)
    out_dir = os.path.expanduser(args.out_dir)
    ensure_dir(out_dir)

    categories = load_categories_from_train_json(train_json)
    print(f"[loaded] categories={len(categories)} from {train_json}")

    # 캐시 파일(중단/재시작 대비)
    cache_path = os.path.join(out_dir, "taxon_cache_by_id.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_by_id = json.load(f)
        # json은 key가 string으로 들어오므로 int->str 통일
        cache_by_id = {str(k): v for k, v in cache_by_id.items()}
        print(f"[cache] loaded {len(cache_by_id)} taxa from {cache_path}")
    else:
        cache_by_id = {}

    cat_meta_rows = []
    cat_to_family = {}  # category_id -> family info

    for idx, cat in enumerate(categories):
        cat_id = int(cat["id"])
        cat_name = str(cat["name"])
        cat_super = str(cat["supercategory"])

        key = str(cat_id)
        taxon = cache_by_id.get(key)

        if taxon is None:
            # 1) id로 먼저 시도
            taxon = None
            for t in range(args.max_retry):
                taxon = get_taxon_by_id(cat_id, include_ancestors=True)
                if taxon is not None:
                    break
                time.sleep(args.sleep * (2 ** t))

            # 2) 그래도 없으면 name으로 검색 fallback
            if taxon is None:
                for t in range(args.max_retry):
                    taxon = search_taxon_by_name(cat_name)
                    if taxon is not None:
                        # ancestors가 없을 수 있으니 id로 다시 ancestors 포함해서 재조회 시도
                        tid = taxon.get("id", None)
                        if tid is not None:
                            taxon2 = get_taxon_by_id(int(tid), include_ancestors=True)
                            if taxon2 is not None:
                                taxon = taxon2
                        break
                    time.sleep(args.sleep * (2 ** t))

            cache_by_id[key] = taxon  # None도 저장해서 반복 호출 방지
            if (idx + 1) % 200 == 0:
                dump_json(cache_path, cache_by_id)
            time.sleep(args.sleep)

        fam_id, fam_name = (None, None)
        if isinstance(taxon, dict):
            fam_id, fam_name = find_family_from_taxon(taxon)

        row = {
            "category_id": cat_id,
            "category_name": cat_name,
            "category_supercategory": cat_super,
            "taxon_found": bool(isinstance(taxon, dict)),
            "taxon_id": (taxon.get("id") if isinstance(taxon, dict) else None),
            "taxon_rank": (taxon.get("rank") if isinstance(taxon, dict) else None),
            "taxon_name": (taxon.get("name") if isinstance(taxon, dict) else None),
            "family_id": fam_id,
            "family_name": fam_name,
        }
        cat_meta_rows.append(row)
        cat_to_family[str(cat_id)] = {"family_id": fam_id, "family_name": fam_name}

        if (idx + 1) % 200 == 0:
            print(f"[progress] {idx+1}/{len(categories)}")

    # 최종 저장
    dump_json(cache_path, cache_by_id)
    dump_json(os.path.join(out_dir, "category_taxonomy_meta.json"), cat_meta_rows)
    dump_json(os.path.join(out_dir, "category_to_family.json"), cat_to_family)

    # 간단 요약
    n_found = sum(1 for r in cat_meta_rows if r["taxon_found"])
    n_fam = sum(1 for r in cat_meta_rows if r["family_id"] is not None)
    fam_set = set(r["family_id"] for r in cat_meta_rows if r["family_id"] is not None)
    print(f"[done] taxon_found={n_found}/{len(categories)} | family_mapped={n_fam}/{len(categories)} | #families={len(fam_set)}")
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
