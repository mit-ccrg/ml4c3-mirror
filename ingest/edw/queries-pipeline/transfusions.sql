SELECT
    t3.PatientIdentityID as MRN
    ,t2.PatientEncounterID as CSN
    ,t1.OrderID
    ,t1.BloodCodingSystemDSC
    ,t1.BloodProductCD
    ,t1.BloodUnitID
    ,t1.BloodStartInstantDTS
    ,t1.BloodEndInstantDTS
    -- ,t1.AdministeredProcedureTypeDSC
    ,t2.ProcedureDSC
    ,t2.OrderStatusDSC
    -- ,t2.StartDTS
    -- ,t2.EndDTS
FROM
    [Epic].[Orders].[BloodAdministration_MGH] t1
    left join Epic.Orders.Procedure_MGH t2 on (t1.OrderID = t2.OrderProcedureID)
    inner join Epic.Patient.Identity_MGH t3 on t2.PatientID = t3.PatientID
WHERE
    t2.PatientEncounterID in ({}) and t3.IdentityTypeID = 67
    -- AND t2.OrderStatus = Completed
    -- AND t1.BloodStartInstantDTS is not null
    -- AND t1.BloodEndInstantDTS is not null
