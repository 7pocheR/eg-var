/-
  EG-VAR Ontology: Domain-Specific Types and Records

  Defines concrete ontologies for the weather and finance evaluation domains,
  following specs/query-formalization-rules.md.
-/
import EGVar.Basic

-- ============================================================
-- § 1. Base Ontology (Σ₀)
-- ============================================================

/-- Location identifier (resolved from city names by trusted lookup table). -/
axiom Location : Type

/-- Temperature in decidegrees Celsius (integer, avoids float issues).
    22.5°C = 225 decidegrees. -/
abbrev Temperature := Int

/-- Temperature unit carried for rendering/debugging; comparisons use the
    canonical decidegrees-Celsius value. -/
inductive TempUnit where
  | celsius
  | fahrenheit
  | kelvin
  deriving DecidableEq, Repr

/-- Canonical temperature quantity: store original reading plus canonical value
    used by kernel-checked comparisons. -/
structure TempQuantity where
  canonicalDeciC : Temperature
  originalDeci   : Int
  unit           : TempUnit
  deriving Repr

/-- Percentage (0-1000, i.e., tenths of a percent for precision). -/
abbrev Percentage := Nat

/-- Price in cents (avoids float issues). $178.50 = 17850 cents. -/
abbrev Price := Nat

/-- Currency identifier. -/
inductive Currency where
  | USD | EUR | GBP | JPY | CHF | CNY
  deriving DecidableEq, Repr

/-- Stock ticker. -/
axiom Ticker : Type

/-- Ratio (stored as numerator × 10000 for fixed-point arithmetic).
    1.2345 = 12345. -/
abbrev Ratio := Nat

/-- Canonical money quantity for cross-currency comparison.
    `canonicalMinor` is the amount converted into a canonical currency
    (Phase 1 uses USD cents) using a specific FX observation snapshot. -/
structure MoneyQuantity where
  canonicalMinor : Int
  originalMinor  : Int
  originalCurr   : Currency
  canonicalCurr  : Currency
  fxTime         : Timestamp
  deriving Repr

/-- Absolute difference on canonical temperatures. -/
def tempGap (x y : Temperature) : Nat :=
  Int.natAbs (x - y)

/-- Numeric disagreement threshold for temperatures. -/
def TempDisagreement (x y : Temperature) (minGap : Nat) : Prop :=
  minGap ≤ tempGap x y

instance {x y : Temperature} {minGap : Nat} : Decidable (TempDisagreement x y minGap) := by
  unfold TempDisagreement tempGap
  infer_instance

/-- Canonical comparisons operate on normalized quantities, not rendered units. -/
def TempGT (x y : TempQuantity) : Prop :=
  x.canonicalDeciC > y.canonicalDeciC

def MoneyGT (x y : MoneyQuantity) : Prop :=
  x.canonicalMinor > y.canonicalMinor

-- ============================================================
-- § 2. Weather Ontology (Σ_weather)
-- ============================================================

/-- Weather world predicates. -/
axiom TempAt      : Location → Timestamp → Temperature → WProp
axiom HumidityAt  : Location → Timestamp → Percentage → WProp
axiom WindSpeedAt : Location → Timestamp → Nat → WProp  -- in 0.1 m/s

/-- Weather observation record (structured payload from weather API). -/
structure WeatherObs where
  loc      : Location
  time     : Timestamp
  temp     : Temperature
  humidity : Percentage
  wind     : Nat

/-- Weather tool identifier. -/
axiom weatherTool : ToolId

/-- Weather query: request weather for a location. -/
structure WeatherQuery where
  loc : Location

/-- Wire weatherTool to EG-VAR tool infrastructure (M6). -/
axiom weatherTool_query_eq : Query_T weatherTool = WeatherQuery
axiom weatherTool_payload_eq : ∀ q : Query_T weatherTool,
  Payload_T weatherTool q = WeatherObs

-- ============================================================
-- § 3. Finance Ontology (Σ_finance)
-- ============================================================

/-- Finance world predicates. -/
axiom PriceAt      : Ticker → Timestamp → Price → Currency → WProp
axiom MarketCapAt  : Ticker → Timestamp → Price → Currency → WProp
axiom PERatioAt    : Ticker → Timestamp → Ratio → WProp
axiom ExchangeRate : Currency → Currency → Timestamp → Ratio → WProp

/-- Price quote record (from finance API). -/
structure PriceQuote where
  ticker   : Ticker
  time     : Timestamp
  price    : Price
  currency : Currency

/-- FX quote record. -/
structure FXQuote where
  base     : Currency
  quote    : Currency
  time     : Timestamp
  rate     : Ratio

/-- Finance tool identifier. -/
axiom financeTool : ToolId

/-- Wire financeTool to EG-VAR tool infrastructure (M6). -/
-- Finance has two query types (price vs FX); modeled as a sum
inductive FinanceQuery where
  | priceQuery (ticker : Ticker)
  | fxQuery (base quote : Currency)

axiom financeTool_query_eq : Query_T financeTool = FinanceQuery

/-- Finance payload depends on query kind. -/
axiom financeTool_payload_eq : ∀ q : Query_T financeTool,
  Payload_T financeTool q = match (cast (financeTool_query_eq ▸ rfl) q : FinanceQuery) with
    | .priceQuery _ => PriceQuote
    | .fxQuery _ _ => FXQuote

-- ============================================================
-- § 4. Entity Lookup Tables (Trusted)
-- ============================================================

/-- City name → Location resolution. Trusted, deterministic.
    Partial function: returns none if city not found. -/
axiom resolveCity : String → Option Location

/-- Ticker string → Ticker resolution. -/
axiom resolveTicker : String → Option Ticker

-- ============================================================
-- § 5. Concrete Claim Tags (H2 fix: enumerable claims per tool)
--
-- Each tool's claims are a finite enum, not abstract.
-- This supports "for each claim in payload" minting and frozen fixtures.
-- ============================================================

/-- Weather tool produces exactly these claims from one observation. -/
inductive WeatherClaimKind where
  | temp       -- TempAt loc time v
  | humidity   -- HumidityAt loc time h
  | windSpeed  -- WindSpeedAt loc time w
  deriving DecidableEq, Repr

/-- Finance tool claim kinds. -/
inductive FinanceClaimKind where
  | price      -- PriceAt ticker time price currency
  | exchangeRate -- ExchangeRate base quote time rate
  deriving DecidableEq, Repr

-- ============================================================
-- § 6. Wiring Claim Enums to ClaimTag_T (H2 completion)
--
-- Connect the concrete enums to the abstract ClaimTag_T bridge
-- so the renderer can target specific claims by kind.
-- ============================================================

/-- Weather: ClaimTag_T for weatherTool IS WeatherClaimKind. -/
axiom weatherTool_claimTag_eq : ∀ (q : Query_T weatherTool) (p : Payload_T weatherTool q),
  ClaimTag_T weatherTool q p = WeatherClaimKind

-- NOTE: Per-claim-kind axioms (weatherClaim_temp, etc.) were removed after
-- review red team review (Finding 2): they quantified over arbitrary loc/t/v
-- without tying to the decoded payload, making them unsound.
-- The generated-script pipeline uses per-query *_shape axioms instead,
-- which are emitted by the trusted runtime from actual decoded payloads.
-- A computable claim decoder is Phase 2 work.

/-- Finance: ClaimTag_T for financeTool IS FinanceClaimKind. -/
axiom financeTool_claimTag_eq : ∀ (q : Query_T financeTool) (p : Payload_T financeTool q),
  ClaimTag_T financeTool q p = FinanceClaimKind

-- The runtime uses these enums to select which claim to mint.
-- The generated script specifies the claim kind, and the renderer
-- emits the appropriate ClaimTag_T + claimOfTag invocation.
-- This makes the claim set enumerable and frozen-fixture-friendly.
