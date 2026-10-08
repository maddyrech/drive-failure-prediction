"""Run the whole pipeline in order.

Usage:
    python -m src.run_pipeline              # real data: download, then everything
    python -m src.run_pipeline --sample     # quick test on simulated data
    python -m src.run_pipeline --skip-download
    python -m src.run_pipeline --azure      # also upload data to Azure Blob Storage
"""
import argparse
import time

from src import (
    download,
    load_postgres,
    make_sample_data,
    run_sql,
    spark_etl,
    stats_analysis,
    train_model,
    write_summary,
)


def step(name, fn, *args):
    print(f"\n=== {name} ===")
    started = time.time()
    fn(*args)
    print(f"--- {name} done in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="use simulated data (for testing only)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--azure", action="store_true", help="upload raw and processed data to Azure")
    args = parser.parse_args()

    if args.sample:
        step("Simulated sample data", make_sample_data.simulate, 3000)
    elif not args.skip_download:
        step("Download Backblaze data", download.main)

    step("Spark pipeline", spark_etl.main)
    step("Statistics", stats_analysis.main)
    step("Model training", train_model.main)
    step("Load PostgreSQL", load_postgres.main)
    step("SQL analysis", run_sql.main)
    step("Results summary", write_summary.main)

    if args.azure:
        from src import upload_to_azure

        step("Upload to Azure", upload_to_azure.main, "all")

    if args.sample:
        print("\nReminder: these results come from SIMULATED data. Don't report them anywhere.")
    print("\nAll done. Start the dashboard with: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
