import json
import os
from pathlib import Path

import requests

collection_id = "21bbfaec-6958-46bc-b1cd-1535752f6304"

domain_name = "cellxgene.cziscience.com"
site_url = f"https://{domain_name}"
api_url_base = f"https://api.{domain_name}"

# Create output directory
output_dir = Path("data/visium/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading collection {collection_id}...")

# Get collection metadata
collection_path = f"/curation/v1/collections/{collection_id}"
collection_url = f"{api_url_base}{collection_path}"
res = requests.get(url=collection_url)
res.raise_for_status()
collection_data = res.json()

print(f"Collection: {collection_data.get('name', 'Unknown')}")
print(f"Description: {collection_data.get('description', 'No description')}")

# Save collection metadata
with open(output_dir / "collection_metadata.json", "w") as f:
    json.dump(collection_data, f, indent=2)

# Download datasets
datasets = collection_data.get("datasets", [])
print(f"Found {len(datasets)} datasets in collection")

for i, dataset in enumerate(datasets, 1):
    dataset_id = dataset.get("dataset_id")
    dataset_name = dataset.get("name", f"dataset_{i}")
    dataset_title = dataset.get("title", dataset_name)

    # Get additional metadata for descriptive filename
    tissue = dataset.get("tissue", [])
    disease = dataset.get("disease", [])
    assay = dataset.get("assay", [])
    cell_type = dataset.get("cell_type", [])

    print(f"\nDownloading dataset {i}/{len(datasets)}: {dataset_title}")
    print(f"Dataset ID: {dataset_id}")
    if tissue:
        print(f"Tissue: {', '.join([t.get('label', str(t)) for t in tissue[:3]])}")
    if disease:
        print(f"Disease: {', '.join([d.get('label', str(d)) for d in disease[:3]])}")
    if assay:
        print(f"Assay: {', '.join([a.get('label', str(a)) for a in assay[:3]])}")

    # Download H5AD file if available from dataset assets
    assets = dataset.get("assets", [])
    h5ad_assets = [asset for asset in assets if asset.get("filetype") == "H5AD"]

    if h5ad_assets:
        h5ad_asset = h5ad_assets[0]  # Take first H5AD file
        download_url = h5ad_asset.get("url")

        # Create descriptive filename
        filename_parts = []

        # Add tissue info
        if tissue:
            tissue_names = [t.get("label", str(t)) for t in tissue[:2]]
            filename_parts.extend(tissue_names)

        # Add disease info
        if disease:
            disease_names = [d.get("label", str(d)) for d in disease[:2]]
            filename_parts.extend(disease_names)

        # Add assay info
        if assay:
            assay_names = [a.get("label", str(a)) for a in assay[:2]]
            filename_parts.extend(assay_names)

        # Clean up filename parts and create filename
        clean_parts = []
        for part in filename_parts:
            # Remove special characters and spaces, convert to lowercase
            clean_part = "".join(c.lower() if c.isalnum() else "_" for c in str(part))
            clean_part = "_".join(clean_part.split("_")[:3])  # Limit words
            if clean_part and clean_part not in clean_parts:
                clean_parts.append(clean_part)

        # Create final filename
        if clean_parts:
            filename = f"{'_'.join(clean_parts[:4])}_{dataset_id[:8]}.h5ad"
        else:
            filename = f"{dataset_name.replace(' ', '_').lower()}_{dataset_id[:8]}.h5ad"

        print(f"  Downloading {filename}...")

        # Download file
        file_res = requests.get(download_url, stream=True)
        file_res.raise_for_status()

        file_path = output_dir / filename
        with open(file_path, "wb") as f:
            for chunk in file_res.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Saved to: {file_path}")
        print(f"  File size: {file_path.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        print(f"  No H5AD files found for dataset {dataset_name}")

print(f"\nDownload complete! Files saved to: {output_dir.absolute()}")
