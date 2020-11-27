--- Labs query for COVID data ---
SELECT 'null',
    t3.PatientIdentityID
    ,t1.[PatientEncounterID]
    ,t1.[ProcedureDSC]
    ,t2.[ComponentID]
    ,t4.ComponentNM
    ,t4.ComponentCommonNM
    ,t2.ResultTXT
    ,t4.LOINCTXT
    ,t1.[OrderStatusDSC]
    ,t2.[ReferenceRangeUnitCD]
    ,t2.[ResultStatusDSC]
    ,t5.[SpecimenTakenTimeDTS]
FROM
    [Epic].[Orders].[Procedure_MGH] t1
    inner join Epic.Orders.Result_MGH t2 on t1.OrderProcedureID=t2.OrderProcedureID
    inner join Epic.Patient.Identity_MGH t3 on t3.PatientID=t1.PatientID and t3.IdentityTypeID = 67
    inner join epic.Reference.Component t4 on t2.ComponentID=t4.ComponentID
    inner join Epic.Orders.Procedure2_MGH t5 on t1.OrderProcedureID = t5.OrderProcedureID
WHERE
    t1.PatientEncounterID in ()





/***
LOINC table (for reference)

D-dimer
troponin
crp
ferritin 2276-4
esr 4537-7
CK IN SERUM OR PLASMA	2157-6
CK-MB IN SERUM OR PLASMA	13969-1
TROPONIN-T IN SERUM OR PLASMA	6598-7
TROPONIN-I IN SERUM OR PLASMA	10839-9
ldh 2532-0

CMP:
GLUCOSE IN SERUM OR PLASMA	2345-7
POTASSIUM IN SERUM OR PLASMA	2823-3
SODIUM IN SERUM OR PLASMA	2951-2
calcium 17861-6
co2 in serum or plasma 2028-9
CHLORIDE	2075-0
Bicarbonate 1963-8
anion gap in serum or plasma 33037-3
BUN 3094-0
CREATININE IN SERUM OR PLASMA	2160-0
albumin 1751-7
total protein 2885-2
globulin 10834-0
Alkaline phosphatase 6768-6
alanine aminotransferase 1742-6
aspartate aminotransferase 1920-8
bilirubin (total) 1975-2

CBC:
abs lymphocyte count
RBC	789-8
NRBC 19048-8
WBC	6690-2
MCV	787-2
HCT	4544-3
HGB IN BLOOD	718-7
PLT	777-3
MCH 785-6
MCHC 786-4
RDW 788-0
MPV 776-5
***/


/***
d['BaseExcess']= ['11555-0','1925-7']
d['HCO3']=  ['1959-6','1960-4']
d['FiO2']= ['3151-8']
d['pH']= ['11558-4']
d['PaCO2']=['2019-8']
d['Glucose']= ['2345-7', '2341-6', '2339-0','41651-1']
d['Potassium']= ['2823-3','6298-4','32713-0']
d['Calcium'] =['17861-6','41650-3']
d['WBC']= ['6690-2']
d['Hgb']=  ['718-7','30313-1']
d['Hct']= ['4544-3','32354-3','20570-8']
d['Creatinine']= ['2160-0']
d['BUN']= ['3094-0']
d['AST']= ['1920-8']
d['Chloride']= ['2075-0','41650-3']
d['Magnesium']= ['2601-3']
d['Phosphate']= ['2777-1'] # phosphorus
d['PTT']= ['3173-2']
d['Fibrinogen']=  ['3255-7']
d['Alkalinephos']=  ['6768-6']
d['Bilirubin_direct']= ['1968-7']
d['Lactate']= ['2524-7','2518-9','32693-4','2519-7']
d['SaO2']=['2713-6','2708-6','28642-7']
d['Bilirubin_total']=  ['1975-2']
d['TroponinI']=   ['6598-7']
d['Platelets']= ['777-3']
 ***/
