SELECT DISTINCT
    t4.PatientIdentityID as MRN
    ,t6.LinkCSNID as PatientEncounterID
    ,t6.PatientEncounterID as SurgeryEncounterID
    --,t1.PatientClassDSC
    ,t7.DepartmentDSC
    ,t1.ProcedureNM
    ,t3.BeginDTS
    ,t3.EndDTS
    ,t5.ProcedureStartDTS
    ,t5.ProcedureCompletedDTS
    ,t3.ProgressDSC
    --,t3.CaseID
FROM
    [Epic].[Surgery].[Procedure_MGH] t1
    inner join [Epic].[Surgery].[CaseAllProcedure_MGH] t2 on (t1.ProcedureID = t2.ProcedureID)
    inner join [Epic].[Surgery].[Case_MGH] t3 on (t2.CaseID = t3.CaseID)
    inner join [Epic].[Patient].[Identity_MGH] t4 on (t3.PatientID = t4.PatientID)
    left join [Epic].[Surgery].[LogTimingEvent_MGH] t5 on (t3.LogID = t5.LogID)
    inner join [Epic].[Surgery].[AdmissionLink_MGH] t6 on (t3.LogID = t6.LogID)
    inner join [Epic].[Encounter].[ADT_MGH] t7 on (t6.LinkCSNID = t7.PatientEncounterID )
    inner join [Epic].[Reference].[Department] t8 on (t7.DepartmentID = t8.DepartmentID)
WHERE
    t4.IdentityTypeID = 67
    and t1.ProcedureNM in
        (
        'CORONARY ARTERY BYPASS GRAFT'
        )
    and t8.RevenueLocationID = '1002001'
ORDER BY
    t4.PatientIdentityID, t6.LinkCSNID, t1.ProcedureNM
