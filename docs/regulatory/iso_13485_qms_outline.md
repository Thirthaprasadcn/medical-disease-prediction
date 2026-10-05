# ISO 13485 Quality Management System — Outline

This document scaffolds the QMS artifacts required for an industrial medical imaging SaMD program.

## 1. Quality manual

- Scope: design, development, verification, validation, release, and post-market monitoring of Medical Imaging Report.
- Exclusions: sterile manufacturing, implantable hardware (N/A for pure software).

## 2. Core procedures to maintain

| Procedure | Purpose |
|---|---|
| Document control | Versioned SOPs, requirements, and design outputs |
| Risk management interface | Links to ISO 14971 file |
| Design & development | Requirements → architecture → implementation → review |
| Software configuration management | Git tags, model registry, release notes |
| Verification & validation | Protocol, datasets, acceptance criteria |
| CAPA | Corrective/preventive actions from incidents & reviews |
| Supplier control | Dataset licenses, cloud/GPU vendors, open-source components |
| Post-market surveillance | Performance drift, complaint handling |

## 3. Design outputs already in this repository

- Pipeline stages: ingestion → preprocessing → inference → report → clinician review
- Model registry + validation gate (`backend/models/registry/`)
- Audit log for model promotion/rollback
- FHIR export for interoperability evidence
- Persistent clinician feedback export

## 4. Next QMS milestones

1. Appoint Management Representative / Quality lead.
2. Freeze software requirements specification (SRS) from this outline + intended use.
3. Establish design history file (DHF) index.
4. Define release checklist tied to registry-approved models only.
