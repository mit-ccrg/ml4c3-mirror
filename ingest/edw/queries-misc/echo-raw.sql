SELECT DISTINCT
    t3.PatientIdentityID as mrn
    ,t1.PatientID as zid
    ,t1.PatientEncounterID as csn
    ,t1.ProcedureID
    ,t1.ProcedureDSC
    ,t1.OrderProcedureID
    ,t1.OrderDTS
    ,t1.StartDTS
    ,t1.EndDTS
    ,t1.ResultDTS
    ,t1.OrderTypeDSC
    ,t1.OrderDisplayNM
    ,t2.ComponentID
    ,t4.ComponentNM
    ,t4.ComponentCommonNM
    ,t2.ResultTXT
    ,t2.ResultValueNBR
    ,t1.LabStatusDSC
    ,t1.OrderStatusDSC
    ,t2.ReferenceRangeUnitCD
    ,t2.DataTypeDSC
FROM
    Epic.Orders.Procedure_MGH t1
    INNER JOIN Epic.Orders.Result_MGH t2 ON t1.OrderProcedureID=t2.OrderProcedureID
    INNER JOIN Epic.Patient.Identity_MGH t3 ON t3.PatientID=t1.PatientID AND t3.IdentityTypeID=67
    INNER JOIN Epic.Reference.Component t4 ON t2.ComponentID=t4.ComponentID
    INNER JOIN Epic.Orders.Procedure2_MGH t5 ON t1.OrderProcedureID = t5.OrderProcedureID
WHERE
    t1.ProcedureDSC IN ('ECHO TTE', 'ECHOCARDIOGRAM COMPLETE WITH BUBBLE STUDY', 'ECHOCARDIOGRAM PHARMACOLOGICAL STRESS TEST', 'ADULT ECHO TTE', 'ECHOCARDIOGRAM DOPPLER', 'Cardiothoracic Point of Care US', 'US CARDIOTHORACIC POINT OF CARE ', 'OVER READ ADULT ECHO TTE OUTSIDE', 'OUTSIDE ECHO EXAM', 'ECHO 2D IMAGING ONLY TTE', 'ECHO LIMITED TTE', 'ECHOCARDIOGRAM STRESS TEST', 'ECHOCARDIOGRAM LIMITED', 'ECHO COMPLETE TTE WITH AGITATED SALINE', 'ECHO FETAL', 'ECHOCARDIOGRAM 3D COMPLETE', 'ECHOCARDIOGRAM 2D LIMITED', 'ECHOCARDIOGRAM 2D COMPLETE', 'ECHO LIMITED TTE WITH AGITATED SALINE', 'ECHOCARDIOGRAM TRANSTHORACIC')
