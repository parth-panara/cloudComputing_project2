#/bin/bash
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8090
export SPARK_LOG_DIR=/opt/spark/logs
export SPARK_MASTER_LOG=/opt/spark/logs/spark-master.out
export JAVA_HOME=/usr/bin/java
sudo mkdir -p $SPARK_LOG_DIR
sudo touch $SPARK_MASTER_LOG
sudo ln -sf /dev/stdout $SPARK_MASTER_LOG
export SPARK_MASTER_HOST=`hostname`
cd /opt/spark/bin &&
sudo ./spark-class org.apache.spark.deploy.master.Master --ip
$SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port
$SPARK_MASTER_WEBUI_PORT >> $SPARK_MASTER_LOG