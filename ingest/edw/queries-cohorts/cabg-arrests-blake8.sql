(
SELECT DISTINCT
    t4.PatientIdentityID as MRN
    ,t6.LinkCSNID as PatientEncounterID
    ,t7.DepartmentDSC
    --,t1.ProcedureNM
    --,t3.BeginDTS
    --,t3.EndDTS
    --,t3.ProgressDSC
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
    and t8.RevenueLocationID = '1002001' -- MGH Main Campus
)
UNION
(SELECT DISTINCT
    t3.PatientIdentityID as MRN
    ,t2.PatientEncounterID
    ,t1.DepartmentDSC
    --,t1.EventNM
    --,t1.EventDTS
FROM
    [Epic].[Clinical].[EDEvent_MGH] t1
    join [Epic].[Clinical].[EDPatient_MGH] t2 on (t1.EventID = t2.EventID)
    join [Epic].[Patient].[Identity_MGH] t3   on (t2.PatientID = t3.PatientID)
    --join [Epic].[Reference].[Department] t4   on (t1.DepartmentID = t4.DepartmentID)
WHERE
    t1.EventDTS > '2016-04-02 00:00:00.0000000'
    and t1.EventTypeCD in
        (
        '34380'  -- Code Start
        ,'34370' -- Rapid Response Start
        )
    and t1.DepartmentID in
        (
        10020010634  -- MGH ELLISON 8 CARDSURG
        ,10020010624 -- MGH ELLISON 10 STP DWN
        ,10020010625 -- MGH ELLISON 11 CARD\INT
        ,10020010635 -- MGH ELLISON 9 MED\CCU
        ,10020010623 -- MGH BLAKE 8 CARD SICU
        )
    and t3.IdentityTypeID = 67
    --and t4.RevenueLocationID = '1002001' -- MGH Main Campus
)
UNION
(SELECT DISTINCT
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID
    ,t1.DepartmentDSC
FROM
    Epic.Encounter.ADT_MGH t1
    left join Epic.Patient.Identity_MGH t2 on (t1.PatientID=t2.PatientID)
WHERE
    t2.IdentityTypeID = 67
    and t1.DepartmentID = '10020010623' -- MGH BLAKE 8 CARD SICU
)
