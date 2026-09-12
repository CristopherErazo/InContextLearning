#!/usr/bin/env bash
# cluster_status.sh -- one-shot, read-only Slurm diagnostic.
#
# Answers "why is my job still PD and roughly how long will it sit there":
# your jobs + their pending reason and estimated start, the load of every
# partition you are allowed to submit to, the queue ahead of you, and your
# account's fairshare / QoS limits / budget.
#
# Usage:  bash shell/cluster_status.sh              # all your jobs
#         bash shell/cluster_status.sh 1234567      # focus on one job id
#         bash shell/cluster_status.sh --all        # don't filter partitions by account
#
# Runs only squeue / scontrol / sinfo / sshare / sacctmgr / saldo queries.
# Works from any directory.

set -uo pipefail

JOBID=""
SHOW_ALL_PARTS=0
for a in "$@"; do
  case "$a" in
    --all|-a)  SHOW_ALL_PARTS=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    [0-9]*)    JOBID="${a%%.*}" ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

command -v squeue >/dev/null 2>&1 || { echo "squeue not found -- not on a Slurm login node?" >&2; exit 1; }

TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP"' EXIT

ME=${USER:-$(id -un)}
ACCOUNT=${SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-}}
if [ -z "$ACCOUNT" ] && command -v sacctmgr >/dev/null 2>&1; then
  ACCOUNT=$(sacctmgr -nP show assoc user="$ME" format=Account 2>/dev/null | sort -u | head -1)
fi
MYGROUPS=$(id -Gn 2>/dev/null | tr ' ' ',')

SQ="$TMP/squeue"; NODES="$TMP/nodes"; PARTS="$TMP/parts"; STARTS="$TMP/starts"
# jobid|partition|qos|state|priority|nodes|cpus|tres-per-node|user|account|reason|starttime|used|left|name
squeue -h -a -o '%i|%P|%q|%T|%Q|%D|%C|%b|%u|%a|%r|%S|%M|%L|%j' >"$SQ"    2>/dev/null
scontrol -o show nodes                                          >"$NODES" 2>/dev/null
scontrol -o show partition                                      >"$PARTS" 2>/dev/null
squeue -h --start -u "$ME" -o '%i|%S|%N'                        >"$STARTS" 2>/dev/null

hr()  { printf '%s\n' "--------------------------------------------------------------------------------"; }
hdr() { printf '\n== %s\n' "$1"; hr; }

printf '\n%s   user=%s   account=%s   cluster=%s\n' \
  "$(date '+%F %T')" "$ME" "${ACCOUNT:-<unknown>}" "$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{gsub(/ /,"",$2);print $2}')"

# ---------------------------------------------------------------- partitions
# Partitions you may submit to: AllowAccounts/DenyAccounts/AllowGroups vs you.
allowed_parts() {
  [ -s "$PARTS" ] || { awk -F'|' '{print $2}' "$SQ" | tr ',' '\n' | sort -u; return; }
  awk -v acct="$ACCOUNT" -v grps="$MYGROUPS" -v showall="$SHOW_ALL_PARTS" '
    function has(list, item,   n,a,i) {
      if (list=="" || list=="ALL" || list=="(null)") return 1
      n=split(list,a,","); for(i=1;i<=n;i++) if (a[i]==item) return 1
      return 0 }
    function hasany(list, csv,   n,a,i) {
      if (list=="" || list=="ALL" || list=="(null)") return 1
      n=split(csv,a,","); for(i=1;i<=n;i++) if (has(list,a[i])) return 1
      return 0 }
    {
      name=""; allowa="ALL"; denya=""; allowg="ALL"
      n=split($0,t,/[ \t]+/)
      for(i=1;i<=n;i++){
             if (t[i] ~ /^PartitionName=/)  name=substr(t[i],15)
        else if (t[i] ~ /^AllowAccounts=/) allowa=substr(t[i],15)
        else if (t[i] ~ /^DenyAccounts=/)   denya=substr(t[i],14)
        else if (t[i] ~ /^AllowGroups=/)   allowg=substr(t[i],13)
      }
      if (name=="") next
      if (showall==1) { print name; next }
      if (acct!="") {
        if (!has(allowa,acct)) next
        if (denya!="" && denya!="(null)" && has(denya,acct)) next
      }
      if (!hasany(allowg,grps)) next
      print name
    }' "$PARTS"
}

MYPARTS=$(awk -F'|' -v me="$ME" '$9==me{print $2}' "$SQ" | tr ',' '\n' | sort -u)
PLIST=$(printf '%s\n%s\n' "$(allowed_parts)" "$MYPARTS" | sed '/^$/d' | sort -u | paste -sd, -)
[ -n "$PLIST" ] || PLIST=$(awk -F'|' '{print $2}' "$SQ" | tr ',' '\n' | sort -u | paste -sd, -)

# ------------------------------------------------------------------ my jobs
hdr "YOUR JOBS"
if ! awk -F'|' -v me="$ME" '$9==me{f=1} END{exit !f}' "$SQ"; then
  echo "  (no jobs in the queue)"
else
  printf '  %-10s %-18s %-9s %-5s %5s %4s %-22s %s\n' JOBID PARTITION STATE NODES PRIO GPU REASON "TIME(used/left)"
  awk -F'|' -v me="$ME" -v only="$JOBID" '
    function gpn(s,  x){ x=s; if (x ~ /gpu/) { sub(/.*gpu[^0-9]*/,"",x); sub(/[^0-9].*/,"",x); if (x!="") return x+0 } return 0 }
    $9==me && (only=="" || $1==only) {
      printf "  %-10s %-18s %-9s %5s %5s %4s %-22s %s/%s\n", $1,$2,$4,$6,$5,gpn($8)*$6,$11,$13,$14 }' "$SQ"
fi

# pending-job detail: rank in queue, estimated start, reason in plain words
awk -F'|' -v me="$ME" -v only="$JOBID" '$9==me && $4=="PENDING" && (only=="" || $1==only){print $1"|"$2"|"$5"|"$11}' "$SQ" |
while IFS='|' read -r jid jparts prio reason; do
  printf '\n  job %s  (%s)\n' "$jid" "$jparts"
  read -r ahead aheadnodes <<<"$(awk -F'|' -v me="$jid" -v prio="$prio" -v parts="$jparts" '
      function intersects(a,b,  n,x,m,y,i,k){ n=split(a,x,","); m=split(b,y,",");
        for(i=1;i<=n;i++) for(k=1;k<=m;k++) if (x[i]==y[k]) return 1; return 0 }
      $4=="PENDING" && $1!=me && intersects($2,parts) && ($5+0)>(prio+0) { c++; nd+=$6 }
      END{ printf "%d %d", c+0, nd+0 }' "$SQ")"
  printf '    priority       : %s  (%s pending jobs outrank you, asking %s nodes in total)\n' "$prio" "$ahead" "$aheadnodes"
  est=$(awk -F'|' -v j="$jid" '$1==j{print $2" on "$3}' "$STARTS")
  printf '    slurm estimate : %s\n' "${est:-N/A (backfill has not scheduled it yet)}"
  case "$reason" in
    Priority)            why="higher-priority jobs are ahead of you; wait or lower your --time to backfill" ;;
    Resources)           why="YOU ARE NEXT -- just waiting for nodes to free up" ;;
    ReqNodeNotAvail*)    why="requested nodes unavailable, usually a reservation/maintenance window ahead of your time limit" ;;
    QOSMaxJobsPerUserLimit|QOSMaxJobsPerAccountLimit|AssocMaxJobsLimit)
                         why="you/your account hit the max concurrent job count for this QoS -- nothing to do but wait" ;;
    *GrpBillingMinutes*|*GrpBillingRunMinutes*|*GrpTRESRunMins*)
                         why="account budget or running-billing cap is exhausted; jobs release as running ones finish" ;;
    Dependency)          why="waiting on another job you declared a dependency on" ;;
    JobHeldUser|JobHeldAdmin) why="job is HELD -- scontrol release $jid" ;;
    PartitionTimeLimit)  why="your --time exceeds the partition limit; this will never start" ;;
    BeginTime)           why="you asked for a future start time (--begin)" ;;
    *)                   why="" ;;
  esac
  printf '    reason         : %s%s\n' "$reason" "${why:+  -- $why}"
done

# ----------------------------------------------------------- partition load
hdr "PARTITION LOAD  (partitions you can submit to)"
printf '  %-20s %6s %6s %6s %6s %6s | %7s %7s %7s | %5s %5s %8s\n' \
  PARTITION NODES IDLE MIXED ALLOC DOWN GPUtot GPUused GPUfree RUN PEND "PEND-GPU"
if [ -s "$NODES" ]; then
  awk -v want=",$PLIST," '
    function gpus(s){ if (match(s,/gres\/gpu=[0-9]+/)) return substr(s,RSTART+9,RLENGTH-9)+0; return 0 }
    {
      name=""; state=""; parts=""; cfg=""; al=""
      n=split($0,t,/[ \t]+/)
      for(i=1;i<=n;i++){
             if (t[i] ~ /^NodeName=/)   name=substr(t[i],10)
        else if (t[i] ~ /^State=/)     state=substr(t[i],7)
        else if (t[i] ~ /^Partitions=/) parts=substr(t[i],12)
        else if (t[i] ~ /^CfgTRES=/)     cfg=substr(t[i],9)
        else if (t[i] ~ /^AllocTRES=/)    al=substr(t[i],11)
      }
      if (parts=="") next
      base=state; sub(/\+.*/,"",base)
      # DRAIN/DOWN/MAINT/RESERVED nodes accept no new work, whatever they run now
      up = (state ~ /DRAIN|DOWN|FAIL|MAINT|INVAL|POWER|UNK|NO_RESPOND|RESERVED/) ? 0 : 1
      g=gpus(cfg); ga=gpus(al)
      np=split(parts,pp,",")
      for(j=1;j<=np;j++){
        p=pp[j]; if (index(want, "," p ",")==0) continue
        tot[p]++
        if (!up)                      dn[p]++
        else if (base=="IDLE")        id[p]++
        else if (base=="MIXED")       mx[p]++
        else if (base=="ALLOCATED")   ac[p]++
        else                          dn[p]++
        if (up) { gt[p]+=g; gu[p]+=ga } 
      }
    }
    END{ for (p in tot) printf "%s|%d|%d|%d|%d|%d|%d|%d|%d\n", p,tot[p],id[p]+0,mx[p]+0,ac[p]+0,dn[p]+0,gt[p]+0,gu[p]+0,gt[p]-gu[p] }
  ' "$NODES" | sort > "$TMP/nodestats"
else
  # fallback when scontrol show nodes is unavailable
  sinfo -h -o '%R|%D|%T' 2>/dev/null | awk -F'|' -v want=",$PLIST," '
    index(want,","$1",")>0 { tot[$1]+=$2
      s=tolower($3)
      if (s ~ /^idle/) id[$1]+=$2; else if (s ~ /^mix/) mx[$1]+=$2
      else if (s ~ /^alloc/) ac[$1]+=$2; else dn[$1]+=$2 }
    END{ for (p in tot) printf "%s|%d|%d|%d|%d|%d|%d|%d|%d\n", p,tot[p],id[p]+0,mx[p]+0,ac[p]+0,dn[p]+0,0,0,0 }' | sort > "$TMP/nodestats"
fi

awk -F'|' -v want=",$PLIST," '
  function gpn(s,  x){ x=s; if (x ~ /gpu/) { sub(/.*gpu[^0-9]*/,"",x); sub(/[^0-9].*/,"",x); if (x!="") return x+0 } return 0 }
  { np=split($2,pp,",")
    for(j=1;j<=np;j++){ p=pp[j]; if (index(want,","p",")==0) continue
      if ($4=="RUNNING") r[p]++
      else if ($4=="PENDING") { q[p]++; pg[p]+=gpn($8)*$6 } } }
  END{ for (p in r) seen[p]; for (p in q) seen[p]
       for (p in seen) printf "%s|%d|%d|%d\n", p, r[p]+0, q[p]+0, pg[p]+0 }' "$SQ" | sort > "$TMP/jobstats"

join -t'|' -a1 -e0 -o '1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.2,2.3,2.4' \
     "$TMP/nodestats" "$TMP/jobstats" 2>/dev/null |
  awk -F'|' '{ printf "  %-20s %6s %6s %6s %6s %6s | %7s %7s %7s | %5s %5s %8s\n",
               $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12 }'

# --------------------------------------------------------------- my account
hdr "ACCOUNT / LIMITS"
if [ -n "$ACCOUNT" ]; then
  awk -F'|' -v acct="$ACCOUNT" -v me="$ME" '
    function gpn(s,  x){ x=s; if (x ~ /gpu/) { sub(/.*gpu[^0-9]*/,"",x); sub(/[^0-9].*/,"",x); if (x!="") return x+0 } return 0 }
    $10==acct { if ($4=="RUNNING") { ar++; ag+=gpn($8)*$6; if($9==me){mr++; mg+=gpn($8)*$6} }
                else if ($4=="PENDING") { ap++; if($9==me) mp++ } }
    END{ printf "  account %s : %d running (%d GPUs), %d pending\n", acct, ar+0, ag+0, ap+0
         printf "  you        : %d running (%d GPUs), %d pending\n", mr+0, mg+0, mp+0 }' "$SQ"
fi

if command -v sshare >/dev/null 2>&1; then
  echo
  echo "  fairshare (lower FairShare = you have over-consumed => lower priority):"
  sshare -U -u "$ME" -P -n -o Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare 2>/dev/null |
    awk -F'|' '{printf "    %-16s %-12s shares=%-8s norm=%-10s usage=%-12s eff=%-10s fairshare=%s\n",$1,$2,$3,$4,$5,$6,$7}'
fi

MYQOS=$(awk -F'|' -v me="$ME" '$9==me{print $3}' "$SQ" | sort -u | paste -sd, -)
if [ -n "$MYQOS" ] && command -v sacctmgr >/dev/null 2>&1; then
  echo
  echo "  QoS limits for: $MYQOS"
  sacctmgr -nP show qos where name="$MYQOS" \
    format=Name,Priority,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES 2>/dev/null |
    awk -F'|' '{printf "    %-16s prio=%-8s maxwall=%-10s maxjobs/user=%-6s maxsubmit/user=%-6s maxtres/user=%-20s grptres=%s\n",$1,$2,$3,$4,$5,$6,$7}'
fi

if command -v saldo >/dev/null 2>&1; then
  echo
  echo "  budget (saldo -b):"
  saldo -b 2>/dev/null | sed 's/^/    /'
fi

echo
echo "  tip: a shorter --time backfills sooner; 'scontrol show job <id>' has the full picture."
echo
