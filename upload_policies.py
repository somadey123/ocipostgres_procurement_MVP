import os
from pathlib import Path

import oci
from dotenv import load_dotenv

ROOT = Path(__file__).parent
POLICY_DIR = ROOT / "data" / "synthetic" / "policies"

load_dotenv()

def main():
    config = oci.config.from_file(
        file_location=os.environ["OCI_CONFIG_FILE"],
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
    )
    client = oci.object_storage.ObjectStorageClient(config)
    namespace = client.get_namespace().data
    bucket = os.environ["OCI_BUCKET"]

    for file_path in POLICY_DIR.glob("*.md"):
        with file_path.open("rb") as f:
            client.put_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=f"policies/{file_path.name}",
                put_object_body=f.read(),
            )
        print(f"Uploaded {file_path.name}")

if __name__ == "__main__":
    main()