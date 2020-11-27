SELECT DISTINCT
    t3.PatientIdentityID   as MRN
    ,t2.PatientEncounterID
    ,t2.PatientID
    ,t1.DepartmentDSC
    ,t1.EventNM
    -- ,t1.EventID
    -- ,t1.EventTypeCD
    -- ,t2.EventTypeID
    ,t1.EventDTS
    ,t1.EventCommentTXT
FROM
    [Epic].[Clinical].[EDEvent_MGH] t1
    join [Epic].[Clinical].[EDPatient_MGH] t2 on (t1.EventID = t2.EventID)
    join [Epic].[Patient].[Identity_MGH] t3   on (t2.PatientID = t3.PatientID)
    join [Epic].[Reference].[Department] t4   on (t1.DepartmentID = t4.DepartmentID)
WHERE
    t1.EventDTS > '2016-04-02 00:00:00.0000000'
    and t1.EventTypeCD in
        (
        '34380'  -- Code Start
        ,'34370' -- Rapid Response Start
        )
    and t4.RevenueLocationID = '1002001' --MGH Main Campus
    and t3.IdentityTypeID = 67
ORDER BY
    t3.PatientIdentityID, t2.PatientEncounterID, t1.EventDTS
