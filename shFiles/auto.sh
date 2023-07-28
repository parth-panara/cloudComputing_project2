#/bin/bash
sudo apt-get update
sudo apt-get install -y curl vim wget software-properties-common
ssh net-tools ca-certificates
sudo apt install -y default-jre
sudo wget --no-verbose -O apache-spark.tgz
"https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-b
in-hadoop2.7.tgz"
sudo mkdir -p /opt/spark
sudo tar -xf apache-spark.tgz -C /opt/spark --strip-components=1
sudo rm apache-spark.tgz