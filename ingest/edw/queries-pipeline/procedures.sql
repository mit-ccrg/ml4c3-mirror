SELECT
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID as CSN
    ,t1.ProcedureDSC
    ,t1.StartDTS
    ,t1.EndDTS
    ,t1.OrderStatusDSC
    ,t1.OrderTypeDSC
    ,t1.ReasonForCancellationDSC
    ,t1.OrderDisplayNM
    ,t1.OrderPriorityDSC
    ,t1.StandingIntervalDSC
    ,t1.ScheduledStartDTS
FROM
    [Epic].[Orders].[Procedure_MGH] t1
    inner join Epic.Patient.Identity_MGH t2 on t2.PatientID = t1.PatientID
WHERE
    t1.[PatientEncounterID] in ({}) and t2.IdentityTypeID = 67
    AND t1.OrderTypeDSC LIKE '%DIALYSIS%'
    -- AND t1.OrderStatus = Completed
    -- AND t1.StartDTS is not null
    -- AND t1.EndDTS is not null
