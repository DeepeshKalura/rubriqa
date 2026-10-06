"""API benchmark against a running app: pagination latency, payload sizes,
dashboard query time. Also reports container memory via `docker stats`.

Usage:
    python -m scripts.bench_api [--base-url http://127.0.0.1:8090]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx


def timed_get(client: httpx.Client, url: str, rounds: int = 5) -> list[dict]:
    results = []
    for _ in range(rounds):
        started = time.perf_counter()
        response = client.get(url)
        elapsed = (time.perf_counter() - started) * 1000
        results.append(
            {
                "url": url,
                "status": response.status_code,
                "ms": round(elapsed, 1),
                "bytes": len(response.content),
            }
        )
    return results


def docker_stats() -> list[dict]:
    try:
        output = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}",
                "exam-eval-web-1",
                "exam-eval-worker-1",
                "exam-eval-db-1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows = []
        for line in output.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) == 3:
                rows.append({"container": parts[0], "memory": parts[1], "cpu": parts[2]})
        return rows
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)[:200]}]


def main(base_url: str) -> None:
    base = base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=30)
    try:
        stats = client.get("/api/stats").json()
    except Exception as exc:  # noqa: BLE001
        print(f"App not reachable at {base}: {exc}")
        raise SystemExit(2)

    print("Dataset:")
    print(
        f"  students={stats['students']:,}  submissions={stats['submissions']:,}  "
        f"documents={stats['documents']:,}  pages={stats['pages']:,}  "
        f"answers={stats['answers']:,}"
    )

    exam_id = None
    exams = client.get("/api/stats").json()
    # find an exam with results via the results page
    exams_response = client.get("/exams")
    import re

    match = re.search(r"/exams/(\d+)", exams_response.text)
    if match:
        exam_id = int(match.group(1))

    endpoints = [
        "/api/stats",
        "/students?page=1&per_page=25",
        "/students?page=100&per_page=25",
        "/students?page=800&per_page=25",
        "/students?q=SCALE0000&per_page=25",
        "/results?page=1",
        "/results?page=400",
        "/jobs?page=1",
    ]
    if exam_id:
        endpoints.extend(
            [
                f"/exams/{exam_id}/results?page=1&per_page=25",
                f"/exams/{exam_id}/results?page=400&per_page=25",
            ]
        )

    all_results: dict[str, dict] = {}
    print("\nEndpoint latency (5 rounds each):")
    print(f"  {'endpoint':58s} {'p50 ms':>8s} {'max ms':>8s} {'KB':>7s} {'status':>6s}")
    for endpoint in endpoints:
        runs = timed_get(client, endpoint)
        latencies = [run["ms"] for run in runs]
        all_results[endpoint] = {
            "p50_ms": round(statistics.median(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "bytes": runs[0]["bytes"],
            "status": runs[0]["status"],
        }
        print(
            f"  {endpoint:58s} {all_results[endpoint]['p50_ms']:8.1f} "
            f"{all_results[endpoint]['max_ms']:8.1f} {runs[0]['bytes'] / 1024:7.1f} "
            f"{runs[0]['status']:6d}"
        )

    print("\nContainer memory:")
    containers = docker_stats()
    for row in containers:
        if "error" in row:
            print(f"  {row['error']}")
        else:
            print(f"  {row['container']:26s} {row['memory']:>18s}  cpu {row['cpu']:>7s}")

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": stats,
        "endpoints": all_results,
        "containers": containers,
        "note": (
            "Synthetic rows only: this measures the data layer and UI pagination, "
            "not OCR/LLM throughput on 20k real exams."
        ),
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = reports / f"scale_api_{stamp}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nReport: {path}")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://web:8000")
    args = parser.parse_args()
    main(args.base_url)


if __name__ == "__main__":
    run()
