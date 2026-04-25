-- EG-VAR Lean Library
-- Root module — imports all components

-- Core infrastructure (unchanged)
import EGVar.Basic
import EGVar.Safety

-- Legacy domain-specific ontology (weather/finance — kept for backward compat)
import EGVar.Ontology
import EGVar.Rules

-- World-level ontology (Phase 2 — tool-agnostic, per ADR-005)
import EGVar.WorldOntology
import EGVar.WorldRules

-- L1 primitives — table storage representation (Phase B, per R10)
import EGVar.TableL1

-- Proof tactics library (reusable patterns for the solver)
import EGVar.Tactics
