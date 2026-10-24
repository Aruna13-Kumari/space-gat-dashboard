import requests
from pathlib import Path


# ============================================================
# PROJECT DATA FOLDER
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASETS USED IN THE PROJECT
# ============================================================

datasets = {
    "raw_tle_active.txt": "active",
    "raw_tle_asat.txt": "1999-025",
    "raw_tle_iridium.txt": "iridium-33-debris",
    "raw_tle_cosmos.txt": "cosmos-2251-debris",
}


# ============================================================
# DOWNLOAD TLE
# ============================================================

def download_tle(group, output_file):

    url = (
        "https://celestrak.org/NORAD/elements/"
        f"gp.php?GROUP={group}&FORMAT=tle"
    )

    print(f"\nDownloading: {group}")

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30
    )

    response.raise_for_status()

    text = response.text.strip()

    # --------------------------------------------------------
    # Validate that CelesTrak actually returned TLE data
    # --------------------------------------------------------

    if (
        "Invalid query" in text
        or "not found" in text.lower()
        or not text
    ):
        raise RuntimeError(
            f"CelesTrak did not return valid TLE data "
            f"for group: {group}\n\n"
            f"Response:\n{text[:500]}"
        )

    # Basic TLE validation:
    # Every object should have:
    # NAME ,1, 2
    
    
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    valid_count = 0

    for i in range(len(lines) - 2):

        if (
            lines[i + 1].startswith("1 ")
            and
            lines[i + 2].startswith("2 ")
        ):
            valid_count += 1

    if valid_count == 0:

        raise RuntimeError(
            f"No valid TLE records found for group: {group}"
        )

    output_path = DATA_DIR / output_file

    output_path.write_text(
        text,
        encoding="utf-8"
    )

    print(
        f"Downloaded {output_file} "
        f"({valid_count} TLE records)"
    )


# ============================================================
# DOWNLOAD ALL DATASETS
# ============================================================

successful_files = []

for filename, group in datasets.items():

    try:

        download_tle(
            group,
            filename
        )

        successful_files.append(filename)

    except Exception as e:

        print(
            f"\nWARNING: Could not download "
            f"{group}"
        )

        print(e)


# ============================================================
# COMBINE ONLY VALID DOWNLOADS
# ============================================================

combined_file = DATA_DIR / "raw_tle_all.txt"

with combined_file.open(
    "w",
    encoding="utf-8"
) as outfile:

    for filename in successful_files:

        file_path = DATA_DIR / filename

        outfile.write(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        outfile.write("\n")


# ============================================================
# VALIDATE COMBINED FILE
# ============================================================

lines = [
    line.strip()
    for line in combined_file.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

valid_records = 0

for i in range(len(lines) - 2):

    if (
        lines[i + 1].startswith("1 ")
        and
        lines[i + 2].startswith("2 ")
    ):
        valid_records += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)

print(
    f"\nDatasets successfully downloaded: "
    f"{len(successful_files)}/{len(datasets)}"
)

print(
    f"Valid TLE records: {valid_records}"
)

print(
    f"Combined file: {combined_file}"
)

print("=" * 60)