from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as date_type
from datetime import datetime, timedelta
from pathlib import Path


EXPECTED_COLUMNS = [
    "date",
    "timestamp",
    "update_timestamp",
    "station_id",
    "station_name",
    "station_device_id",
    "location_longitude",
    "location_latitude",
    "reading_update_timestamp",
    "reading_value",
    "reading_type",
    "reading_unit",
]


def parse_epoch_seconds(value: bytes) -> int:
    """Parse the fixed ISO-8601 Singapore timestamps used by the files."""

    text = value.decode("ascii")
    return int(datetime.fromisoformat(text).timestamp())


def scan_file(path_text: str) -> dict:
    path = Path(path_text)
    null_counts = Counter()
    station_rows = Counter()
    station_positive_rows = Counter()
    station_rainfall = defaultdict(float)
    station_names = defaultdict(set)
    station_locations = defaultdict(set)
    station_devices = defaultdict(set)
    reading_types = Counter()
    reading_units = Counter()
    dates = set()
    timezones = Counter()

    rows = malformed_rows = invalid_values = 0
    zero_rows = positive_rows = negative_rows = 0
    duplicate_station_timestamp = date_mismatches = 0
    timestamp_regressions = station_time_regressions = 0
    min_value = None
    max_value = None
    rainfall_sum = 0.0
    min_timestamp = max_timestamp = None
    timestamp_groups = 0

    current_timestamp = None
    current_epoch = None
    previous_timestamp_epoch = None
    timestamp_station_masks = {}
    station_bit_indexes = {}

    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8-sig").split(",")

        for raw_line in handle:
            rows += 1
            fields = raw_line.rstrip(b"\r\n").split(b",")
            if len(fields) != len(EXPECTED_COLUMNS):
                malformed_rows += 1
                continue

            for index, field in enumerate(fields):
                if not field:
                    null_counts[EXPECTED_COLUMNS[index]] += 1

            date, timestamp = fields[0], fields[1]
            station_id, station_name, device_id = fields[3], fields[4], fields[5]
            longitude, latitude = fields[6], fields[7]
            value_raw, reading_type, reading_unit = fields[9], fields[10], fields[11]

            if timestamp != current_timestamp:
                timestamp_groups += 1
                current_timestamp = timestamp
                try:
                    current_epoch = parse_epoch_seconds(timestamp)
                    if previous_timestamp_epoch is not None:
                        gap = current_epoch - previous_timestamp_epoch
                        if gap < 0:
                            timestamp_regressions += 1
                    previous_timestamp_epoch = current_epoch
                except (ValueError, UnicodeDecodeError, OverflowError):
                    current_epoch = None

            station_bit_index = station_bit_indexes.setdefault(
                station_id, len(station_bit_indexes)
            )
            station_bit = 1 << station_bit_index
            prior_mask = timestamp_station_masks.get(timestamp, 0)
            if prior_mask & station_bit:
                duplicate_station_timestamp += 1
            timestamp_station_masks[timestamp] = prior_mask | station_bit

            date_text = date.decode("ascii", errors="replace")
            timestamp_text = timestamp.decode("ascii", errors="replace")
            dates.add(date_text)
            if date != timestamp[:10]:
                date_mismatches += 1
            if len(timestamp) >= 6:
                timezones[timestamp[-6:].decode("ascii", errors="replace")] += 1
            min_timestamp = timestamp_text if min_timestamp is None else min(min_timestamp, timestamp_text)
            max_timestamp = timestamp_text if max_timestamp is None else max(max_timestamp, timestamp_text)

            station_text = station_id.decode("utf-8", errors="replace")
            station_rows[station_text] += 1
            station_names[station_text].add(station_name.decode("utf-8", errors="replace"))
            station_devices[station_text].add(device_id.decode("utf-8", errors="replace"))
            station_locations[station_text].add(
                (longitude.decode("ascii", errors="replace"), latitude.decode("ascii", errors="replace"))
            )
            reading_types[reading_type.decode("utf-8", errors="replace")] += 1
            reading_units[reading_unit.decode("utf-8", errors="replace")] += 1

            try:
                value = float(value_raw)
            except ValueError:
                invalid_values += 1
                continue

            rainfall_sum += value
            min_value = value if min_value is None else min(min_value, value)
            max_value = value if max_value is None else max(max_value, value)
            if value == 0:
                zero_rows += 1
            elif value > 0:
                positive_rows += 1
                station_positive_rows[station_text] += 1
            else:
                negative_rows += 1
            station_rainfall[station_text] += value

    timestamp_gaps = Counter()
    station_gaps = Counter()
    last_station_epoch = [None] * len(station_bit_indexes)
    sorted_timestamp_masks = sorted(
        (parse_epoch_seconds(timestamp), mask)
        for timestamp, mask in timestamp_station_masks.items()
    )
    previous_epoch = None
    for epoch, mask in sorted_timestamp_masks:
        if previous_epoch is not None:
            timestamp_gaps[str(epoch - previous_epoch)] += 1
        previous_epoch = epoch
        remaining = mask
        while remaining:
            low_bit = remaining & -remaining
            bit_index = low_bit.bit_length() - 1
            prior_epoch = last_station_epoch[bit_index]
            if prior_epoch is not None:
                station_gaps[str(epoch - prior_epoch)] += 1
            last_station_epoch[bit_index] = epoch
            remaining ^= low_bit

    station_counts_per_timestamp = [mask.bit_count() for mask in timestamp_station_masks.values()]
    stations_per_timestamp_min = min(station_counts_per_timestamp, default=None)
    stations_per_timestamp_max = max(station_counts_per_timestamp, default=0)

    year = int(path.stem[-4:])
    cursor = date_type(year, 1, 1)
    year_end = date_type(year, 12, 31)
    missing_dates = []
    while cursor <= year_end:
        cursor_text = cursor.isoformat()
        if cursor_text not in dates:
            missing_dates.append(cursor_text)
        cursor += timedelta(days=1)

    def sorted_counter(counter: Counter) -> dict:
        return dict(sorted(counter.items(), key=lambda item: int(item[0])))

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "header": header,
        "header_matches_expected": header == EXPECTED_COLUMNS,
        "rows": rows,
        "malformed_rows": malformed_rows,
        "null_counts": dict(null_counts),
        "invalid_reading_values": invalid_values,
        "zero_rows": zero_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "rainfall_sum_mm": round(rainfall_sum, 3),
        "min_reading_mm": min_value,
        "max_reading_mm": max_value,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "date_count": len(dates),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "date_timestamp_mismatches": date_mismatches,
        "timezone_counts": dict(timezones),
        "unique_timestamps": len(timestamp_station_masks),
        "timestamp_groups_in_source_order": timestamp_groups,
        "timestamp_gap_seconds": sorted_counter(timestamp_gaps),
        "timestamp_regressions": timestamp_regressions,
        "station_count": len(station_rows),
        "stations_per_timestamp_min": stations_per_timestamp_min,
        "stations_per_timestamp_max": stations_per_timestamp_max,
        "duplicate_station_timestamp_rows": duplicate_station_timestamp,
        "station_gap_seconds": sorted_counter(station_gaps),
        "station_time_regressions": station_time_regressions,
        "missing_dates": missing_dates,
        "reading_types": dict(reading_types),
        "reading_units": dict(reading_units),
        "station_rows": dict(sorted(station_rows.items())),
        "station_positive_rows": dict(sorted(station_positive_rows.items())),
        "station_rainfall_mm": {
            key: round(value, 3) for key, value in sorted(station_rainfall.items())
        },
        "station_name_variants": {
            key: sorted(value) for key, value in sorted(station_names.items()) if len(value) > 1
        },
        "station_device_variants": {
            key: sorted(value) for key, value in sorted(station_devices.items()) if len(value) > 1
        },
        "station_location_variants": {
            key: sorted([list(item) for item in value])
            for key, value in sorted(station_locations.items())
            if len(value) > 1
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("rainfall_20*.csv"))
    if not paths:
        raise SystemExit(f"No annual CSV files found in {args.data_dir}")

    results = []
    if args.workers <= 1:
        for path in paths:
            result = scan_file(str(path))
            results.append(result)
            print(f"Scanned {result['file']}: {result['rows']:,} rows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as executor:
            futures = {executor.submit(scan_file, str(path)): path for path in paths}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Scanned {result['file']}: {result['rows']:,} rows", flush=True)

    results.sort(key=lambda item: item["file"])
    output = args.data_dir / "audit_summary.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
