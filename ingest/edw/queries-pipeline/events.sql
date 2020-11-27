SELECT
    t3.PatientIdentityID   as MRN
    ,t2.PatientEncounterID as CSN
    --,t2.PatientID
    ,t1.DepartmentDSC
    ,t1.EventDTS
    --,t1.EventID
    --,t1.EventTypeCD
    --,t2.EventTypeID
    ,t1.EventNM
    ,t1.EventCommentTXT
    ,t1.EventStatusDSC
FROM
    [Epic].[Clinical].[EDEvent_MGH] t1
    join [Epic].[Clinical].[EDPatient_MGH] t2 on (t1.EventID = t2.EventID)
    join [Epic].[Patient].[Identity_MGH] t3   on t2.PatientID = t3.PatientID
WHERE
    t2.[PatientEncounterID] in ({}) and t3.IdentityTypeID=67
    AND t1.EventTypeCD in (
	'34380'  -- Code Start
	,'34370' -- Rapid Response Start
    )
