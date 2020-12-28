SELECT DISTINCT
    t_proc.PatientID as zid
    ,t_proc.PatientEncounterID as csn
    ,t_mrn.PatientIdentityID as mrn
    ,t_empi.PatientIdentityID as empi
    ,t_proc.OrderProcedureID
    ,t_img.AccessionID
    ,t_proc.ProcedureDSC
    ,t_proc.OrderTypeDSC
    ,t_proc.OrderDisplayNM
    ,t_res.ComponentID
    ,t_comp.ComponentNM
    ,t_comp.ComponentCommonNM
    ,t_res.ResultValueNBR
    ,t_res.ReferenceRangeUnitCD
FROM
    Epic.Orders.Procedure_MGH t_proc
    INNER JOIN Epic.Orders.Result_MGH t_res ON t_proc.OrderProcedureID=t_res.OrderProcedureID
    INNER JOIN Epic.Reference.Component t_comp ON t_res.ComponentID=t_comp.ComponentID
    INNER JOIN Epic.Orders.ImagingAccessionNumber_MGH t_img ON t_img.OrderProcedureID=t_proc.OrderProcedureID
    LEFT JOIN Epic.Patient.Identity_MGH t_mrn ON t_mrn.PatientID=t_proc.PatientID AND t_mrn.IdentityTypeID=67
    LEFT JOIN Epic.Patient.Identity_MGH t_empi ON t_empi.PatientID=t_proc.PatientID AND t_empi.IdentityTypeID=140
WHERE
    t_img.AccessionID IN ('foobar')
