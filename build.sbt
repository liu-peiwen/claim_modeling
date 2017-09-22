name := "claims_modeling"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.6.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.6.0"
libraryDependencies += "mysql" % "mysql-connector-java" % "5.1.38"
resolvers += "Maven Central" at "https://repo1.maven.org/maven2/"
