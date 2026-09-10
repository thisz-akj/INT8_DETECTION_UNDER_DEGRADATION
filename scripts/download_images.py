import os
import sys
import concurrent.futures as cf
import urllib.request

BASE_URL = "http://images.cocodataset.org/val2017/"

def load_list(path):
    items = []
    with open(path) as f:
        for line in f:
            iid, fname = line.strip().split("\t")
            items.append((iid, fname))
    return items

def download_one(args):
    fname, out_dir = args
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return fname, "skip"
    url = BASE_URL + fname
    try:
        urllib.request.urlretrieve(url, out_path)
        return fname, "ok"
    except Exception as e:
        return fname, f"fail: {e}"

def main():
    list_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    items = load_list(list_path)
    print(f"Downloading {len(items)} images to {out_dir} ...")
    fails = []
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for fname, status in ex.map(download_one, [(fn, out_dir) for _, fn in items]):
            if status not in ("ok", "skip"):
                fails.append((fname, status))
    print(f"Done. Failures: {len(fails)}")
    for f in fails[:20]:
        print(f)

if __name__ == "__main__":
    main()
