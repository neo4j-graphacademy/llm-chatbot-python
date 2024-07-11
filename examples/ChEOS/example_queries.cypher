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
