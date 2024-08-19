// delete all relations MEASURED_AT in chunks to avoid mem overflow
MATCH (s:Substance)-[r:MEASURED_AT]->()
CALL {
WITH r
DETACH DELETE r
} IN TRANSACTIONS OF 10000 ROWS

// delete all substance nodes
MATCH (n:Substance)
DETACH DELETE n

// where is Diuron a driver with high driver importance
MATCH (s:Substance {preferredName: 'Diuron'})-[r:IS_DRIVER]->(l:Site)
  WHERE r.driver_importance > 0.8
RETURN s, r, l

// find all substances measured above a threshold in a certain river
MATCH (c:Substance)-[r:MEASURED_AT]->(s:Site {water_body: 'seine'})
  WHERE r.mean_concentration > 0
RETURN c.DTXSID AS DTXSID, c.preferredName AS Name

MATCH (c:Substance)-[r:MEASURED_AT]->(s:Site {country: 'France'})
RETURN DISTINCT c.DTXSID AS DTXSID, c.preferredName AS Name

// find most frequent drivers
MATCH (s:Substance)-[r:IS_DRIVER]->(l:Site)
  WHERE r.driver_importance > 0.8
RETURN DISTINCT s.name, s.DTXSID, r.driver_importance
  ORDER BY r.driver_importance