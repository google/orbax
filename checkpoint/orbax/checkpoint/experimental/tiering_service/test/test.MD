To create cluster that will mount an existing Lustre

```bash
python3 cts_e2e_test_runner.py \
create_demo_cluster \
    --project=orbax-checkpoint \
    --cluster_name=cts-demo-cluster-sm \
    --zone=us-east1-b \
    --instance_type=n4d-standard-2 \
    --num_nodes=3 \
    --network_tier=PREMIUM \
    --luster_instance=lustre-us-east1-b \
    --spot \
    --luster_mount=/lustre
```

To provision a running pod and install source codes and dependency

```bash
python3 cts_e2e_test_runner.py bootstrap \
    --cluster cts-demo-cluster-sm \
    --zone us-east1-b \
    --pod_name cts-demo-runner-pod \
    --github_pr 3482
```

Now you can run this command to verify the successful installation

```
kubectl exec cts-demo-runner-pod -n default -- python3 -m orbax.checkpoint.experimental.tiering_service.test.cts_client_cli --help
```

