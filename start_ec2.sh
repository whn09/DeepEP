# aws ec2 run-instances --region $REGION \
#     --instance-type $INSTANCETYPE \
#     --image-id $AMI --key-name $KEYNAME \
#     --iam-instance-profile "Name=dlami-builder" \
#     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
#     --network-interfaces "NetworkCardIndex=0,DeviceIndex=0,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=1,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=2,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=3,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=4,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=5,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=6,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=7,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=8,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=9,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=10,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=11,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=12,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=13,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=14,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
#      "NetworkCardIndex=15,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa"

export REGION="us-west-2"
export INSTANCETYPE="p5en.48xlarge"
export AMI="ami-0cd047a60c9801486"
export KEYNAME="YOUR_KEYNAME"
export TAG="P5"
export SG="sg-0ecfda87ea8a4a25e"
export SUBNET="subnet-39072272"
aws ec2 run-instances --region $REGION \
    --instance-type $INSTANCETYPE \
    --image-id $AMI --key-name $KEYNAME \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
    --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"YOUR_CapacityReservationId"}}' \
    --instance-market-options '{"MarketType":"capacity-block"}' \
    --metadata-options '{"HttpTokens":"required"}' \
    --count "2" \
    --network-interfaces "NetworkCardIndex=0,DeviceIndex=0,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=1,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=2,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=3,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=4,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=5,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=6,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=7,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=8,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=9,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=10,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=11,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=12,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=13,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=14,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
     "NetworkCardIndex=15,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa"