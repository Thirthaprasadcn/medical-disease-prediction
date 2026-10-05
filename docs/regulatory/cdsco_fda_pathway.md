# CDSCO / FDA Pathway Notes (India + US planning)

**Disclaimer:** Not legal advice. Engage a regulatory consultant before submission.

## Product type

Software as a Medical Device (SaMD) — AI/ML-enabled imaging decision support.

## India (CDSCO)

1. Confirm device classification under Medical Devices Rules (risk-based).
2. Appoint Indian Authorized Agent if applicable.
3. Prepare Device Master File / technical dossier: intended use, software description, V&V, risk file, cybersecurity.
4. Clinical evaluation / performance evaluation aligned to claimed modalities (CT lung nodules, MRI brain tumors).
5. Post-approval PMS plan.

## United States (FDA)

1. Determine pathway (likely 510(k) or De Novo depending on predicates/claims).
2. Align with FDA AI/ML SaMD guiding principles: locked vs predetermined change control plan for model updates.
3. Map model registry + validation gate to change-control evidence.
4. Human factors / usability for clinician review UI.
5. Cybersecurity documentation for networked DICOM/FHIR interfaces.

## Evidence package checklist

- [ ] Intended use frozen
- [ ] Software requirements & architecture
- [ ] Training/validation datasets documented (LIDC-IDRI, BraTS, etc.)
- [ ] Pre-specified metrics and acceptance criteria
- [ ] Clinical reader study protocol
- [ ] Labeling (UI disclaimers, IFU)
- [ ] QMS (ISO 13485) and lifecycle (IEC 62304)
- [ ] Risk management (ISO 14971)
- [ ] Cybersecurity + PHI handling

## Repository mapping

| Evidence need | Location |
|---|---|
| Intended use | `docs/regulatory/intended_use.md` |
| QMS outline | `docs/regulatory/iso_13485_qms_outline.md` |
| Lifecycle | `docs/regulatory/iec_62304_lifecycle.md` |
| Risk | `docs/regulatory/iso_14971_risk.md` |
| Model gate | `backend/models/registry/` |
| Metrics disclosure | report JSON + FHIR extensions |
