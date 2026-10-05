# ISO 14971 Risk Management File (Starter)

## Scope

Risks arising from AI-assisted imaging report software used as clinician decision support.

## Top hazards (initial)

| ID | Hazard | Harm | Initial controls | Residual strategy |
|---|---|---|---|---|
| H1 | False negative (missed lesion) | Delayed care | Sensitivity gates; clinician review mandatory; disclaimer | Clinical validation on target modality |
| H2 | False positive overload | Unnecessary follow-up | Specificity gates; severity labels; review UI | Threshold tuning + feedback loop |
| H3 | Modality mismatch (X-ray model on CT) | Misleading output | Modality router; clinical vs demo mode; registry metadata | Block demo model in clinical deployments |
| H4 | PHI exposure via PACS/logs | Privacy breach | AE titles, local storage controls, audit notes, TLS requirement before production | Institutional security review |
| H5 | Unvalidated model swap | Silent performance drop | Validation gate + checksum + rollback | Dual control approval |
| H6 | Over-trust of prototype | Diagnostic misuse | Persistent UI disclaimer; preliminary FHIR status | Labeling & training |

## Risk acceptability

No residual risk is acceptable for unsupervised diagnostic use. Software remains **adjunct to radiologist judgment** until clearance and clinical evidence support a broader claim.

## Traceability

- Controls implemented in code: disclaimers, registry gate, review persistence, FHIR `preliminary` status.
- Evidence to collect: dataset cards, hold-out metrics, human factors study of review UI.
