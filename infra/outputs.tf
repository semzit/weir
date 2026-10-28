output "region" {
  value = var.region
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "artifact_uri_prefix" {
  value = "s3://${aws_s3_bucket.artifacts.bucket}/runs/"
}

output "compute_environment" {
  value = aws_batch_compute_environment.gpu.name
}

output "job_queue" {
  value = aws_batch_job_queue.gpu.name
}

output "job_queue_arn" {
  value = aws_batch_job_queue.gpu.arn
}

output "job_definition" {
  value = aws_batch_job_definition.train.arn
}

output "cpu_compute_environment" {
  value = aws_batch_compute_environment.cpu.name
}

output "cpu_job_queue" {
  value = aws_batch_job_queue.cpu.name
}

output "cpu_job_definition" {
  value = aws_batch_job_definition.train_cpu.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.batch.name
}

output "image" {
  value = local.image
}
