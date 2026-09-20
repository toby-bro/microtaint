# microtaint NDSS 2027 artifact, run 20260918-121340

mode: quick (reduced corpus, NOT the paper numbers)
engine version: v0.7.2-dirty
MICROTAINT_TAINT_IR=0

avalanche-calibrate      PASS exit 0
avalanche-base64-pinned  PASS exit 0
avalanche-base64-system  PASS exit 0
avalanche-nftables       PASS exit 0
avalanche-siphash        PASS exit 0
rq7-bof                  PASS 1 bof finding(s) detected
rq7-uaf                  PASS 1 uaf finding(s) detected
rq7-sc                   PASS 1 side_channel finding(s) detected
rq7-aiw                  PASS 1 aiw finding(s) detected
rq7-crypto-check         PASS exit 0
rq7-crypto-localise      PASS exit 0
rq7-dns                  PASS exit 0
rq5-ladder               PASS exit 0
rq5-bench                PASS exit 0
rq6-pass1                PASS exit 0
rq6-pass2                PASS exit 0
macros-avalanche         SKIP needs all three avalanche reports
macros-benchmark         SKIP no rq2 report to generate from
