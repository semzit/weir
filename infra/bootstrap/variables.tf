variable "region" {
  type        = string
  description = "AWS region that stores the OpenTofu state bucket."
}

variable "state_bucket_name" {
  type        = string
  description = "Override for the state bucket name; defaults to weir-tofu-state-<account-id>."
  default     = null
}
