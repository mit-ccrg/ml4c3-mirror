SELECT
    t2.PatientIdentityID as MRN
    ,t6.PatientEncounterID
    ,t6.LinkCSNID
    ,t4.ProcedureNM
    --,t1.PatientClassDSC
    ,t1.BeginDTS
    ,t1.EndDTS
    ,t5.ProcedureStartDTS
    ,t5.ProcedureCompletedDTS
    ,t1.ProgressDSC
    --,t1.CaseID
FROM
    [Epic].[Surgery].[Case_MGH] t1
    inner join Epic.Patient.Identity_MGH t2 on t1.PatientID = t2.PatientID
    inner join Epic.Surgery.CaseAllProcedure_MGH t3 on (t1.CaseID = t3.CaseID)
    inner join Epic.Surgery.Procedure_MGH t4 on (t3.ProcedureID = t4.ProcedureID)
    left join Epic.Surgery.LogTimingEvent_MGH t5 on (t1.LogID = t5.LogID)
    inner join Epic.Surgery.AdmissionLink_MGH t6 on (t1.LogID = t6.LogID)
WHERE
    -- t6.PatientEncounterID in ({0})
    t6.LinkCSNID in ({0}) and t2.IdentityTypeID = 67
