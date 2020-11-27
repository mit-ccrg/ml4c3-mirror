SELECT  DISTINCT
	'null',
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID as CSN
    ,t1.ProcedureDSC
    ,t1.StartDTS
    ,t1.EndDTS
FROM
    [Epic].[Orders].[Procedure_MGH] t1
    inner join Epic.Patient.Identity_MGH t2 on (t2.PatientID = t1.PatientID and t2.IdentityTypeID in (67,227))
WHERE
    t1.[PatientEncounterID] in () -- include all (old and new) encounters
AND (t1.ProcedureDSC LIKE '%INTUBATION%' or t1.ProcedureDSC LIKE '%EXTUBATION%' or t1.ProcedureDSC LIKE '%MECHANICAL VENTILATION%')
AND t1.OrderStatusDSC in ('Completed')
AND t1.StartDTS is not null
AND t1.EndDTS is not null
