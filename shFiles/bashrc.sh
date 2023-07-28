function snotebook ()
{
SPARK_PATH=~/opt/spark/spark-3.2.0-bin-hadoop2.7
export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
export PYSPARK_PYTHON=python3
$SPARK_PATH/bin/pyspark --master local[2]
}
