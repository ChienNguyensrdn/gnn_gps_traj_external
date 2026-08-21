import os
from pathlib import Path
import zipfile
import fcntl


class _PlainProgress:
    """Minimal tqdm-compatible fallback; dataset download must not require UI packages."""

    def __init__(self, total=None, initial=0, **_kwargs):
        self.total = total
        self.current = initial

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        if self.total:
            print(f"Downloaded {self.current}/{self.total} bytes")
        else:
            print(f"Downloaded {self.current} bytes")

    def update(self, amount):
        self.current += amount


def _progress_bar(**kwargs):
    try:
        from tqdm import tqdm
        return tqdm(**kwargs)
    except ImportError:
        print("tqdm is unavailable in this Python environment; continuing without a progress bar.")
        return _PlainProgress(**kwargs)

def downlad_model(model_name="google/gemma-2-2b-it", model_path="/you_local_path/gemma-2-2b-it/", download_tool="modelscope"):
    DOWNLOAD_TOOL = download_tool
    if DOWNLOAD_TOOL == "hf":
        from huggingface_hub import snapshot_download
        from config import PROXY
        print("using huggingface downloader, please use proxy")

        os.environ["http_proxy"] = PROXY
        os.environ["https_proxy"] = PROXY

        snapshot_download(model_name, local_dir=model_path, resume_download=True, local_dir_use_symlinks=False, max_workers=1)

        del os.environ["http_proxy"]
        del os.environ["https_proxy"]
    elif DOWNLOAD_TOOL == "hf-mirror":
        # https://hf-mirror.com/
        pass
    elif DOWNLOAD_TOOL == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download
        model_dir = snapshot_download(model_name, cache_dir=model_path)


def download_data(data_name="www19", use_proxy=True):
    """
        A function to download the dataset from the URL if the dataset is not present in the current directory
        You can also directly download the dataset from the URL and place it in the current directory
        :return: None
    """
    import gdown
    import requests
    from config import WWW2019_DATA_DIR, TIST2015_DATA_DIR, TSMC2014_DATA_DIR, GOWALLA_DATA_DIR, PROXY, DATA_PATH

    dataset_urls_dict = {
        "tsmc2014": 'http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip',
        # 'https://drive.google.com/file/d/0BwrgZ-IdrTotZ0U0ZER2ejI3VVk/view'
        "tist2015": 'https://github.com/tsinghua-fib-lab/AgentMove/releases/download/dataset_tist2015/dataset_tist2015.zip',
        "www2019": 'https://github.com/tsinghua-fib-lab/AgentMove/blob/main/data/www2019_isp_data.zip',
        "gowalla": 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
    }
    dataset_url = dataset_urls_dict[data_name]
    
    if use_proxy:
        os.environ["http_proxy"] = PROXY
        os.environ["https_proxy"] = PROXY
    print('Downloading the dataset {}...'.format(data_name))
    # check if the dataset is already present in the current directory
    if data_name=="tsmc2014" and not os.path.exists(os.path.join(TSMC2014_DATA_DIR, 'dataset_TSMC2014_NYC.txt')):
        r = requests.get(dataset_url)
        with open(os.path.join(DATA_PATH, "dataset_tsmc2014.zip"), 'wb') as f:
            f.write(r.content)
        print('Download complete!')

        print('Extracting the zip folder...')
        os.system('unzip {} -d {}/'.format(os.path.join(DATA_PATH, "dataset_tsmc2014.zip"), DATA_PATH))
        print('Extraction complete!')

        print('Removing the zip folder...')
        os.system('rm {}'.format(os.path.join(DATA_PATH, "dataset_tsmc2014.zip")))
        print('Removal complete!')

    elif data_name=="tist2015" and not os.path.exists(os.path.join(TIST2015_DATA_DIR, "dataset_TIST2015_Checkins.txt")):
        # TIST2015 is hosted as a regular GitHub Release asset, not a Google
        # Drive link. Newer gdown releases removed ``fuzzy`` and should not be
        # used for this URL. Stream with requests so redirects, status errors,
        # progress, and the destination filename are handled explicitly.
        archive = Path(DATA_PATH) / "dataset_tist2015.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        lock_path = archive.with_suffix(".zip.lock")
        with lock_path.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another TIST2015 download is active: {lock_path}") from exc
            downloaded = archive.stat().st_size if archive.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            with requests.get(dataset_url, stream=True, timeout=180, headers=headers) as response:
                response.raise_for_status()
                # A server may ignore Range and return 200. In that case restart
                # safely; append only when it confirms the requested byte offset.
                content_range = response.headers.get("content-range", "")
                resumed = downloaded > 0 and response.status_code == 206 and content_range.startswith(f"bytes {downloaded}-")
                if downloaded and not resumed:
                    downloaded = 0
                remaining = int(response.headers.get("content-length", 0))
                total = downloaded + remaining if remaining else None
                with archive.open("ab" if resumed else "wb") as handle, _progress_bar(
                    total=total, initial=downloaded, unit="B", unit_scale=True, unit_divisor=1024
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            progress_bar.update(len(chunk))
        print('Download complete!')

        print('Extracting the zip folder...')
        with zipfile.ZipFile(archive) as bundle:
            corrupt_member = bundle.testzip()
            if corrupt_member:
                raise zipfile.BadZipFile(f"CRC failure in {corrupt_member}; remove {archive} and retry")
            bundle.extractall(DATA_PATH)
        print('Extraction complete!')

        print('Removing the zip folder...')
        archive.unlink()
        print('Removal complete!')

    elif data_name=="www2019" and not os.path.exists(os.path.join(WWW2019_DATA_DIR, "isp")):
        ## we have uploaded data, no need to download dataset
        # if not os.path.exists(os.path.join(DATA_PATH, "www2019_isp_data.zip")):
        #     if use_proxy:
        #         os.environ["http_proxy"] = PROXY
        #         os.environ["https_proxy"] = PROXY
        #     r = requests.get(dataset_url, stream=True, timeout=180)
        #     file_size = int(r.headers.get('content-length', 0))
        #     with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
        #         with open(os.path.join(DATA_PATH, "www2019_isp_data.zip"), 'wb') as f:
        #             for chunk in r.iter_content(chunk_size=1024):
        #                 f.write(chunk)
        #                 progress_bar.update(len(chunk)) 
        #     print('Download complete!')

        print('Extracting the zip folder...')
        os.system('unzip {} -d {}/'.format(os.path.join(DATA_PATH, "www2019_isp_data.zip"), WWW2019_DATA_DIR))
        print('Extraction complete!')

        os.system('unzip {}/isp.zip -d {}/'.format(WWW2019_DATA_DIR, WWW2019_DATA_DIR))
        os.system('unzip {}/poi.txt.zip -d {}/'.format(WWW2019_DATA_DIR, WWW2019_DATA_DIR))
        os.system('rm {}/isp.zip'.format(WWW2019_DATA_DIR))
        os.system('rm {}/poi.txt.zip'.format(WWW2019_DATA_DIR))

        # print('Removing the zip folder...')
        # os.system('rm {}'.format(os.path.join(DATA_PATH, "www2019_isp_data.zip")))
        # print('Removal complete!')

    elif data_name=="gowalla" and not os.path.exists(os.path.join(GOWALLA_DATA_DIR, "gowalla_totalCheckins.txt")):
        r = requests.get(dataset_url)
        with open(os.path.join(DATA_PATH, "gowalla_totalCheckins.txt.gz"), 'wb') as f:
            f.write(r.content)
        print('Download complete!')

        print('Extracting the zip folder...')
        os.makedirs(GOWALLA_DATA_DIR, exist_ok=True)
        os.system('gunzip {}'.format(os.path.join(DATA_PATH, "gowalla_totalCheckins.txt.gz")))
        print('Extraction complete!')

        print('Move the data to right folder...')
        os.system('mv {} {}'.format(os.path.join(DATA_PATH, "gowalla_totalCheckins.txt"), GOWALLA_DATA_DIR))
    else:
        print('Dataset already present in the current directory!')
    
    if use_proxy:
        del os.environ["http_proxy"]
        del os.environ["https_proxy"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--download_mode', type=str, default="data", choices=["data", "model"])
    parser.add_argument("--data_name", type=str, default="www2019", choices=["tsmc2014", "tist2015", "www2019", "gowalla"])
    parser.add_argument("--use_proxy", action='store_true')
    args = parser.parse_args()

    if args.download_mode == "data":
        download_data(data_name=args.data_name, use_proxy=args.use_proxy)
    else:
        downlad_model(model_name="google/gemma-2-2b-it", model_path="/you_local_path/gemma-2-2b-it/", download_tool="hf")
