# Approach Doc: 002 - Persistent Storage & Experiment Strategy

## 1. Objective
Transform GADS from a system that performs ephemeral file dumps into a high-integrity **Research Ledger** where every execution is immutable, reproducible, and content-addressed.

## 2. Core Principles
- **Immutability**: Once a DAG Run is finished, its parameters, code, and artifacts are sealed. Re-running creates a new Run ID.
- **Content-Addressing**: All blobs are indexed by their SHA-256 hash. This enables cross-project deduplication and instant "dry-run" checks.
- **Contract-First Serving**: The UI never touches the filesystem directly; it requests signed URLs from an abstracted `ArtifactStore`.

## 3. Data Hierarchy

| Level | Maps to | Mutability | Description |
| :--- | :--- | :--- | :--- |
| **Project** | Logical Folder | Name Only | Top-level container (e.g., "Titanic Survival"). |
| **Experiment**| MLflow Experiment | Metadata | A specific hypothesis or line of inquiry. |
| **Run** | **MLflow Run** | **Never** | A single DAG execution. Sealed on completion. |
| **Artifact** | S3/Local Blob | **Never** | The actual CSV, PNG, or MD file. |

## 4. The "Commit" Protocol (Safety)
To prevent the "Data Integrity Gap" (orphaned files during crashes), we use a Two-Phase Commit:

1. **Step 1 (Intent)**: Sandbox notifies Backend: "I am about to write report.pdf." Registry creates a PENDING row.
2. **Step 2 (Staging)**: Sandbox writes to a temporary .staging/ path.
3. **Step 3 (Commit)**: Sandbox notifies Backend of completion + file hash. 
4. **Step 4 (Seal)**: Backend performs an **Atomic Rename** from staging to final path and flips DB status to COMMITTED.

## 5. Evolution Path

### Phase 1: Hybrid Pointer (MVP)
- **Metadata**: PostgreSQL + SQLModel.
- **Storage**: Local Filesystem with an X-Accel-Redirect (Nginx) or X-Sendfile serving layer for efficiency.
- **Registry**: A ProjectFile table that tracks every file's path, hash, and owner task.

### Phase 2: Research Grade (Mid-Term)
- **Tracking**: Full **MLflow** integration. Every project execution becomes an MLflow Run.
- **Storage**: **MinIO / S3**. Artifacts are synced from the Sandbox to object storage.
- **Metrics**: Automated mining of metrics (Accuracy, ROC-AUC) from task stdout via the Synthesizer.

### Phase 3: Enterprise Hub (Scale)
- **Multi-tenancy**: Isolated S3 buckets per organization/team.
- **Versioning**: DVC (Data Version Control) integration for dataset lineage.
- **Audit**: Comprehensive download/read logs for all sensitive artifacts.

## 6. Abstracting the Store
To ensure zero application code changes during the migration from Local to S3, we implement an ArtifactStore protocol:

```python
class ArtifactStore(Protocol):
    def get_upload_url(self, artifact_id: str) -> str: ...
    def get_download_url(self, artifact_id: str, ttl: int) -> str: ...
    def commit(self, artifact_id: str, actual_hash: str) -> bool: ...
```

- **Phase 1 Impl**: Returns a backend-signed local path.
- **Phase 2 Impl**: Returns an S3 pre-signed URL.
