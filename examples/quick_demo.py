"""Run PEAC-Log on a tiny in-memory HDFS-style example."""

from main import PEACLogPipeline, preprocess_logs


def main() -> None:
    raw_logs = [
        "Received block blk_123 from 10.0.0.1:50010",
        "Received block blk_456 from 10.0.0.2:50010",
        "Deleted block blk_123 on /data/node-a",
        "Deleted block blk_456 on /data/node-b",
    ]
    cleaned_logs = preprocess_logs("HDFS", raw_logs)
    pipeline = PEACLogPipeline("HDFS")
    _, event_to_template, _, duration = pipeline.parse(cleaned_logs)

    print("\nCleaned logs:")
    for line in cleaned_logs:
        print(f"  {line}")
    print("\nExtracted templates:")
    for event_id, template in event_to_template.items():
        print(f"  {event_id}: {template}")
    print(f"\nDuration: {duration:.4f}s")


if __name__ == "__main__":
    main()
