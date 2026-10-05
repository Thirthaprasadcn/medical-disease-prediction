# IEC 62304 Software Lifecycle Plan (Draft)

## Safety classification (preliminary)

Pending formal hazard analysis. Working assumption for planning: **Class B** (non-serious injury possible if software fails and clinician over-trusts output). Final class follows ISO 14971 risk file.

## Lifecycle processes

1. **Software development planning** — this document + industrial roadmap phases 0–7.
2. **Requirements analysis** — frozen API contracts in `backend/contracts.py`; DICOM/FHIR/PACS requirements in phase docs.
3. **Architectural design** — FastAPI backend, pipeline modules, registry, SQLite review store, optional PACS SCP.
4. **Detailed design & unit implementation** — modules under `backend/pipeline`, `backend/db`, `backend/pacs`, `backend/models/registry`.
5. **Unit verification** — `backend/tests/` (pytest).
6. **Integration testing** — upload PNG/DICOM → report → FHIR → review persistence.
7. **System testing** — institutional protocol before any clinical pilot.
8. **Release** — only registry-approved models; changelog + checksums.
9. **Maintenance** — rollback via registry; CAPA linkage.

## SOUP (software of unknown provenance)

Track and risk-assess at minimum:

- PyTorch, OpenCV, pydicom, pynetdicom, FastAPI, nnDetection, nnU-Net, SQLite

## Problem resolution

Defects that affect finding generation, report integrity, or PHI handling require:

1. Ticket + severity
2. Risk evaluation
3. Fix + regression tests
4. Controlled release through registry gate when model-affecting
