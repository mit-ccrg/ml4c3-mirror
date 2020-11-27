SELECT
    t3.PatientIdentityID as MRN
    ,t1.PatientID
    ,t1.PatientEncounterID
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
    ,t2.ComponentCommentTXT
    ,t4.ComponentNM
    ,t4.ComponentCommonNM
    ,t2.ResultTXT
    ,t2.ResultValueNBR
    ,t4.LOINCTXT
    ,t1.LabStatusDSC
    ,t1.OrderStatusDSC
    ,t2.ReferenceRangeUnitCD
    ,t2.ResultStatusDSC
    ,t2.LabTechnicianID
    ,t2.DataTypeDSC
    ,t2.ComponentObservedDTS
    ,t5.SpecimenReceivedTimeDTS
    ,t5.SpecimenTakenTimeDTS
    ,t5.SpecimenCommentsTXT
FROM
    Epic.Orders.Procedure_MGH t1
    inner join Epic.Orders.Result_MGH t2 on t1.OrderProcedureID=t2.OrderProcedureID
    inner join Epic.Patient.Identity_MGH t3 on t3.PatientID=t1.PatientID
    inner join Epic.Reference.Component t4 on t2.ComponentID=t4.ComponentID
    inner join Epic.Orders.Procedure2_MGH t5 on t1.OrderProcedureID = t5.OrderProcedureID
WHERE
    t1.PatientEncounterID in ({}) and t3.IdentityTypeID = 67
