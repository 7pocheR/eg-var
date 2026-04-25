import EGVar.Basic
import EGVar.WorldOntology
import EGVar.Tactics
set_option autoImplicit false

-- Claim: what is the average value of the 2001 general election across all regions in italy , the answer be 6.16
-- Goal: Check that the average value of the 2001 general election across all regions is 6.16
-- Verdict attempt: ENTAILED

axiom tableTool : ToolId
axiom statsTool : ToolId

axiom apulia : Entity
axiom calabria : Entity
axiom campania : Entity
axiom lazio : Entity
axiom lombardy : Entity
axiom piedmont : Entity
axiom sicily : Entity
axiom tuscany : Entity
axiom veneto : Entity
axiom «emilia - romagna» : Entity
axiom «2001 general» : Property

-- obs0: piedmont.2001 general = 3.5
axiom obs0_q : Query_T tableTool
axiom obs0_p : Payload_T tableTool obs0_q
axiom obs0_att : Attested_T tableTool obs0_q obs0_p
axiom obs0_tag : ClaimTag_T tableTool obs0_q obs0_p
def obs0_val : Value := Value.str "3.5"
axiom obs0_shape : claimWProp obs0_tag = HasProperty piedmont «2001 general» obs0_val

-- obs1: lombardy.2001 general = 3.4
axiom obs1_q : Query_T tableTool
axiom obs1_p : Payload_T tableTool obs1_q
axiom obs1_att : Attested_T tableTool obs1_q obs1_p
axiom obs1_tag : ClaimTag_T tableTool obs1_q obs1_p
def obs1_val : Value := Value.str "3.4"
axiom obs1_shape : claimWProp obs1_tag = HasProperty lombardy «2001 general» obs1_val

-- obs2: veneto.2001 general = 5.0
axiom obs2_q : Query_T tableTool
axiom obs2_p : Payload_T tableTool obs2_q
axiom obs2_att : Attested_T tableTool obs2_q obs2_p
axiom obs2_tag : ClaimTag_T tableTool obs2_q obs2_p
def obs2_val : Value := Value.int 5
axiom obs2_shape : claimWProp obs2_tag = HasProperty veneto «2001 general» obs2_val

-- obs3: emilia - romagna.2001 general = 3.4
axiom obs3_q : Query_T tableTool
axiom obs3_p : Payload_T tableTool obs3_q
axiom obs3_att : Attested_T tableTool obs3_q obs3_p
axiom obs3_tag : ClaimTag_T tableTool obs3_q obs3_p
def obs3_val : Value := Value.str "3.4"
axiom obs3_shape : claimWProp obs3_tag = HasProperty «emilia - romagna» «2001 general» obs3_val

-- obs4: tuscany.2001 general = 3.3
axiom obs4_q : Query_T tableTool
axiom obs4_p : Payload_T tableTool obs4_q
axiom obs4_att : Attested_T tableTool obs4_q obs4_p
axiom obs4_tag : ClaimTag_T tableTool obs4_q obs4_p
def obs4_val : Value := Value.str "3.3"
axiom obs4_shape : claimWProp obs4_tag = HasProperty tuscany «2001 general» obs4_val

-- obs5: lazio.2001 general = 4.8
axiom obs5_q : Query_T tableTool
axiom obs5_p : Payload_T tableTool obs5_q
axiom obs5_att : Attested_T tableTool obs5_q obs5_p
axiom obs5_tag : ClaimTag_T tableTool obs5_q obs5_p
def obs5_val : Value := Value.str "4.8"
axiom obs5_shape : claimWProp obs5_tag = HasProperty lazio «2001 general» obs5_val

-- obs6: campania.2001 general = 7.5
axiom obs6_q : Query_T tableTool
axiom obs6_p : Payload_T tableTool obs6_q
axiom obs6_att : Attested_T tableTool obs6_q obs6_p
axiom obs6_tag : ClaimTag_T tableTool obs6_q obs6_p
def obs6_val : Value := Value.str "7.5"
axiom obs6_shape : claimWProp obs6_tag = HasProperty campania «2001 general» obs6_val

-- obs7: apulia.2001 general = 6.8
axiom obs7_q : Query_T tableTool
axiom obs7_p : Payload_T tableTool obs7_q
axiom obs7_att : Attested_T tableTool obs7_q obs7_p
axiom obs7_tag : ClaimTag_T tableTool obs7_q obs7_p
def obs7_val : Value := Value.str "6.8"
axiom obs7_shape : claimWProp obs7_tag = HasProperty apulia «2001 general» obs7_val

-- obs8: calabria.2001 general = 9.5
axiom obs8_q : Query_T tableTool
axiom obs8_p : Payload_T tableTool obs8_q
axiom obs8_att : Attested_T tableTool obs8_q obs8_p
axiom obs8_tag : ClaimTag_T tableTool obs8_q obs8_p
def obs8_val : Value := Value.int 95
axiom obs8_shape : claimWProp obs8_tag = HasProperty calabria «2001 general» obs8_val

-- obs9: sicily.2001 general = 14.4
axiom obs9_q : Query_T tableTool
axiom obs9_p : Payload_T tableTool obs9_q
axiom obs9_att : Attested_T tableTool obs9_q obs9_p
axiom obs9_tag : ClaimTag_T tableTool obs9_q obs9_p
def obs9_val : Value := Value.str "14.4"
axiom obs9_shape : claimWProp obs9_tag = HasProperty sicily «2001 general» obs9_val

-- obs10: STATS sum_2001 general = 57
axiom obs10_q : Query_T statsTool
axiom obs10_p : Payload_T statsTool obs10_q
axiom obs10_att : Attested_T statsTool obs10_q obs10_p
axiom obs10_tag : ClaimTag_T statsTool obs10_q obs10_p
def obs10_val : Value := Value.int 57
axiom obs10_shape : claimWProp obs10_tag = SumWhere «2001 general» Cond.all (Value.int 57)

-- obs11: STATS count_2001 general = 10
axiom obs11_q : Query_T statsTool
axiom obs11_p : Payload_T statsTool obs11_q
axiom obs11_att : Attested_T statsTool obs11_q obs11_p
axiom obs11_tag : ClaimTag_T statsTool obs11_q obs11_p
def obs11_val : Value := Value.int 10
axiom obs11_shape : claimWProp obs11_tag = CountWhere «2001 general» Cond.all (Value.int 10)

-- obs12: STATS sum_2001 general = 57
axiom obs12_q : Query_T statsTool
axiom obs12_p : Payload_T statsTool obs12_q
axiom obs12_att : Attested_T statsTool obs12_q obs12_p
axiom obs12_tag : ClaimTag_T statsTool obs12_q obs12_p
def obs12_val : Value := Value.int 57
axiom obs12_shape : claimWProp obs12_tag = SumWhere «2001 general» Cond.all (Value.int 57)

-- obs13: STATS sum_2001 general = 57
axiom obs13_q : Query_T statsTool
axiom obs13_p : Payload_T statsTool obs13_q
axiom obs13_att : Attested_T statsTool obs13_q obs13_p
axiom obs13_tag : ClaimTag_T statsTool obs13_q obs13_p
def obs13_val : Value := Value.int 57
axiom obs13_shape : claimWProp obs13_tag = SumWhere «2001 general» Cond.all (Value.int 57)

-- TRUSTED: type generated by pipeline from formalizer's lean_goal_type
-- Solver cannot alter this type — only provide the tactic proof.
noncomputable def claim_proof : Σ (s : Int), Σ (c : Int), Σ (avg10 : Int), PProd (Evidence Grade.Verified (SumWhere «2001 general» Cond.all (Value.int s))) (PProd (Evidence Grade.Verified (CountWhere «2001 general» Cond.all (Value.int c))) (PLift (avg10 = s * 10000000000 / c))) := by
  exact ⟨57, 10, 57000000000, PProd.mk (by rw [<- obs10_shape]; exact mkVerified obs10_att (claimOfTag obs10_tag)) (PProd.mk (by rw [<- obs11_shape]; exact mkVerified obs11_att (claimOfTag obs11_tag)) (PLift.up (by decide)))⟩

-- Axiom audit:
#print axioms claim_proof
