# Production Deployment Guide

This guide covers deploying CryptoRL to production environments with proper monitoring, security, and scalability.

## Overview

Production deployment involves:
- **Container orchestration** with Docker/Kubernetes
- **High availability** with load balancing
- **Security** with secrets management
- **Monitoring** with metrics and alerting
- **Scaling** based on demand

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                         │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Trading   │  │   Training  │  │   Monitor   │        │
│  │   Service   │  │   Service   │  │   Service   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Database Cluster                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ InfluxDB    │  │ PostgreSQL  │  │   Redis     │        │
│  │   Primary   │  │   Primary   │  │   Cluster   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                Message Queue & Cache                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Kafka     │  │   Celery    │  │   MinIO     │        │
│  │   Cluster   │  │   Workers   │  │   Storage   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Production Deployment Options

### 1. Docker Compose (Single Node)

#### Production Compose File
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  cryptorl-api:
    image: cryptorl:latest
    restart: unless-stopped
    environment:
      - CRYPTORL_ENV=production
      - BINANCE_TESTNET=false
    depends_on:
      - influxdb
      - postgresql
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  influxdb:
    image: influxdb:2.7
    restart: unless-stopped
    environment:
      - INFLUXDB_REPORTING_DISABLED=true
      - INFLUXDB_DB=cryptorl
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"

  postgresql:
    image: postgres:15
    restart: unless-stopped
    environment:
      - POSTGRES_DB=cryptorl
      - POSTGRES_USER=cryptorl
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - cryptorl-api

volumes:
  influxdb_data:
  postgres_data:
  redis_data:
```

#### Deployment Commands
```bash
# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale cryptorl-api=3

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### 2. Kubernetes Deployment

#### Namespace and Secrets
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cryptorl

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cryptorl-secrets
  namespace: cryptorl
type: Opaque
data:
  binance-api-key: <base64-encoded-key>
  binance-secret-key: <base64-encoded-secret>
  postgres-password: <base64-encoded-password>
```

#### Deployment Configuration
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cryptorl-api
  namespace: cryptorl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cryptorl-api
  template:
    metadata:
      labels:
        app: cryptorl-api
    spec:
      containers:
      - name: cryptorl-api
        image: cryptorl:latest
        ports:
        - containerPort: 8000
        env:
        - name: CRYPTORL_ENV
          value: "production"
        - name: BINANCE_API_KEY
          valueFrom:
            secretKeyRef:
              name: cryptorl-secrets
              key: binance-api-key
        - name: BINANCE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: cryptorl-secrets
              key: binance-secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service Configuration
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cryptorl-api-service
  namespace: cryptorl
spec:
  selector:
    app: cryptorl-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cryptorl-api-hpa
  namespace: cryptorl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cryptorl-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. Helm Chart Deployment

#### Chart Structure
```yaml
# helm/cryptorl/values.yaml
replicaCount: 3

image:
  repository: cryptorl
  tag: latest
  pullPolicy: Always

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
  - host: cryptorl.yourdomain.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: cryptorl-tls
    hosts:
    - cryptorl.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    database: cryptorl

redis:
  enabled: true
  auth:
    enabled: true
    password: "changeme"
```

#### Helm Commands
```bash
# Install with Helm
helm install cryptorl ./helm/cryptorl \
  --namespace cryptorl \
  --create-namespace \
  --set image.tag=v1.2.3

# Upgrade deployment
helm upgrade cryptorl ./helm/cryptorl \
  --namespace cryptorl \
  --set image.tag=v1.2.4

# Rollback to previous version
helm rollback cryptorl 1
```

## Environment Configuration

### 1. Production Environment Variables

```bash
# .env.production
CRYPTORL_ENV=production
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://cryptorl:${POSTGRES_PASSWORD}@postgres:5432/cryptorl
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=${INFLUXDB_TOKEN}
REDIS_URL=redis://redis:6379

# Binance
BINANCE_API_KEY=${BINANCE_API_KEY}
BINANCE_SECRET_KEY=${BINANCE_SECRET_KEY}
BINANCE_TESTNET=false

# Security
JWT_SECRET_KEY=${JWT_SECRET_KEY}
API_RATE_LIMIT=100
CORS_ORIGINS=https://cryptorl.yourdomain.com

# Monitoring
SENTRY_DSN=${SENTRY_DSN}
PROMETHEUS_ENABLED=true
GRAFANA_URL=https://grafana.yourdomain.com
```

### 2. Configuration Management

#### Kubernetes ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cryptorl-config
  namespace: cryptorl
data:
  config.yaml: |
    trading:
      max_positions: 5
      max_position_size: 0.2
      stop_loss: 0.05
      take_profit: 0.1
    
    risk:
      max_drawdown: 0.15
      var_threshold: 0.05
      
    monitoring:
      metrics_interval: 60
      alert_webhook: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## Security Configuration

### 1. TLS/SSL Setup

#### Nginx Configuration
```nginx
# nginx/nginx.conf
upstream cryptorl_backend {
    server cryptorl-api:8000;
}

server {
    listen 443 ssl http2;
    server_name cryptorl.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+AES256:ECDHE+AES128:!aNULL:!MD5:!DSS;

    location / {
        proxy_pass http://cryptorl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

#### Cert-Manager Configuration
```yaml
# k8s/cert-manager.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### 2. Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cryptorl-netpol
  namespace: cryptorl
spec:
  podSelector:
    matchLabels:
      app: cryptorl-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: cryptorl
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 8086  # InfluxDB
    - protocol: TCP
      port: 6379  # Redis
```

## Monitoring and Alerting

### 1. Prometheus Configuration

#### Prometheus Setup
```yaml
# k8s/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'cryptorl'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - cryptorl
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

#### ServiceMonitor
```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: cryptorl-monitor
  namespace: cryptorl
spec:
  selector:
    matchLabels:
      app: cryptorl-api
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### 2. Grafana Dashboard

#### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "CryptoRL Production",
    "panels": [
      {
        "title": "API Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, cryptorl_api_response_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Trading PnL",
        "targets": [
          {
            "expr": "cryptorl_pnl_total",
            "legendFormat": "Total PnL"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "cryptorl_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

### 3. Alert Rules

#### Prometheus Alert Rules
```yaml
# k8s/alert-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cryptorl-alerts
  namespace: cryptorl
spec:
  groups:
  - name: cryptorl.rules
    rules:
    - alert: HighErrorRate
      expr: rate(cryptorl_api_errors_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors per second"
    
    - alert: TradingLoss
      expr: cryptorl_pnl_total < -1000
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Significant trading loss"
        description: "PnL is ${{ $value }}"
    
    - alert: ModelPerformance
      expr: cryptorl_model_accuracy < 0.5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Model performance degraded"
        description: "Accuracy is {{ $value }}"
```

## Database Configuration

### 1. PostgreSQL Production Setup

#### StatefulSet Configuration
```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: cryptorl
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: cryptorl
        - name: POSTGRES_USER
          value: cryptorl
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: cryptorl-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### 2. InfluxDB Cluster Setup

```yaml
# k8s/influxdb-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: influxdb
  namespace: cryptorl
spec:
  serviceName: influxdb
  replicas: 1
  selector:
    matchLabels:
      app: influxdb
  template:
    spec:
      containers:
      - name: influxdb
        image: influxdb:2.7
        ports:
        - containerPort: 8086
        env:
        - name: INFLUXDB_REPORTING_DISABLED
          value: "true"
        volumeMounts:
        - name: influxdb-storage
          mountPath: /var/lib/influxdb2
  volumeClaimTemplates:
  - metadata:
      name: influxdb-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi
```

## Backup and Disaster Recovery

### 1. Database Backups

#### PostgreSQL Backup
```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: cryptorl
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h postgres -U cryptorl cryptorl > /backup/cryptorl_$DATE.sql
              aws s3 cp /backup/cryptorl_$DATE.sql s3://cryptorl-backups/postgres/
            env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: cryptorl-secrets
                  key: postgres-password
            volumeMounts:
            - name: backup
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup
            emptyDir: {}
```

### 2. InfluxDB Backup

```yaml
# k8s/influxdb-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: influxdb-backup
  namespace: cryptorl
spec:
  schedule: "0 3 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: influxdb-backup
            image: influxdb:2.7
            command:
            - /bin/bash
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              influx backup /backup/influxdb_$DATE
              tar -czf /backup/influxdb_$DATE.tar.gz /backup/influxdb_$DATE
              aws s3 cp /backup/influxdb_$DATE.tar.gz s3://cryptorl-backups/influxdb/
            volumeMounts:
            - name: backup
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup
            emptyDir: {}
```

## Scaling Strategies

### 1. Horizontal Scaling

#### Pod Autoscaling
```yaml
# k8s/hpa-scaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cryptorl-scaler
  namespace: cryptorl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cryptorl-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 2. Vertical Scaling

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: cryptorl-vpa
  namespace: cryptorl
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cryptorl-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: cryptorl-api
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
```

## Deployment Commands

### 1. Initial Deployment
```bash
# Create namespace
kubectl create namespace cryptorl

# Deploy secrets
kubectl apply -f k8s/secrets.yaml

# Deploy configmaps
kubectl apply -f k8s/configmap.yaml

# Deploy services
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/influxdb-statefulset.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Wait for databases
kubectl wait --for=condition=ready pod -l app=postgres -n cryptorl --timeout=300s
kubectl wait --for=condition=ready pod -l app=influxdb -n cryptorl --timeout=300s

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Deploy monitoring
kubectl apply -f k8s/servicemonitor.yaml
kubectl apply -f k8s/alert-rules.yaml
```

### 2. Rolling Updates
```bash
# Update deployment
kubectl set image deployment/cryptorl-api cryptorl-api=cryptorl:v1.2.4 -n cryptorl

# Monitor rollout
kubectl rollout status deployment/cryptorl-api -n cryptorl

# Rollback if needed
kubectl rollout undo deployment/cryptorl-api -n cryptorl
```

### 3. Health Checks
```bash
# Check pod status
kubectl get pods -n cryptorl

# Check services
kubectl get services -n cryptorl

# Check logs
kubectl logs -f deployment/cryptorl-api -n cryptorl

# Port forward for testing
kubectl port-forward service/cryptorl-api-service 8080:80 -n cryptorl
```

## Production Checklist

### Pre-deployment
- [ ] Test all services locally
- [ ] Validate configuration
- [ ] Check resource requirements
- [ ] Run security scan
- [ ] Performance testing

### Deployment
- [ ] Deploy databases first
- [ ] Verify data persistence
- [ ] Deploy application services
- [ ] Configure monitoring
- [ ] Set up alerting

### Post-deployment
- [ ] Verify all endpoints
- [ ] Test trading functionality
- [ ] Monitor resource usage
- [ ] Validate backup process
- [ ] Document runbooks