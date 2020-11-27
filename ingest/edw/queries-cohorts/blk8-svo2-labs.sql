SELECT DISTINCT
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
    INNER JOIN Epic.Orders.Result_MGH t2 ON t1.OrderProcedureID=t2.OrderProcedureID
    INNER JOIN Epic.Patient.Identity_MGH t3 ON t3.PatientID=t1.PatientID AND t3.IdentityTypeID = 67
    INNER JOIN Epic.Reference.Component t4 ON t2.ComponentID=t4.ComponentID
    INNER JOIN Epic.Orders.Procedure2_MGH t5 ON t1.OrderProcedureID = t5.OrderProcedureID
    INNER JOIN Epic.Encounter.ADT_MGH t6 ON t1.PatientEncounterID = t6.PatientEncounterID
WHERE
    t6.DepartmentID = '10020010623'
    AND (
        LOWER(t4.ComponentCommonNM) IN ('so2-mixed (so2, mixed bld)', 'so2-venous (so2, venous)')
        OR (LOWER(t4.ComponentCommonNM) = 'so2, unspecified' AND LOWER(t5.SpecimenCommentsTXT) LIKE '%venous%')
    )
