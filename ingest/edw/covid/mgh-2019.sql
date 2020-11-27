(select *
from Epic.Patient.Infection_Enterprise
where infectiontypedsc in ('COVID-19')
) a
inner join
(select IdentityTypeID,patientid,PatientIdentityID
from Epic.Patient.Identity_Enterprise
where IdentityTypeID = '67') b on a.patientid = b.patientid
inner join
(select patientencounterid
,patientid
,adtpatientclassificationdsc
,adtpatientstatusdsc
,admitsourcedsc
,adtarrivaldts
,adtarrivalstatusdsc
,hospitaladmitdts
,hospitaldischargedts
,hospitaladmittypedsc
,departmentid
,departmentdsc
,hospitalaccountid
,contactdts
,eddispositiondsc
,eddispositiondts
,eddeparturedts
,inpatientadmitdts
,inpatientadmiteventdts
,emergencyadmitdts
,outpatientadmitdts
from epic.encounter.patientencounterhospital_enterprise
where adtpatientclassificationdsc = 'Inpatient' and hospitaladmitdts >= '2020-01-01') i on a.patientid = i.patientid
