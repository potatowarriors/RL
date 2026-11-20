#!/bin/bash
# Acceptance test for the Backend.AI test session (run as the session user, uid 1100).
# Order matters: each step assumes the previous ones passed, so the first FAIL names one cause.
# usage: acceptance.sh [alpine.sif] [swe.sif]      (steps 7+ are skipped if apptainer is not installed)
ALP=${1:-/home/work/vidsearch/tools/nemo_rl/apptainer_test/alpine.sif}
SWE=${2:-}
EXPECT_PROFILE=${EXPECT_PROFILE:-docker-apptainer}
n_fail=0
ok()   { echo "  PASS  $1"; }
bad()  { echo "  FAIL  $1"; echo "        cause: $2"; n_fail=$((n_fail+1)); }
note() { echo "  NOTE  $1"; }

echo "== 0. what this session was created with"
sec=$(awk '/^Seccomp:/{print $2}' /proc/self/status); cap=$(awk '/^CapBnd:/{print $2}' /proc/self/status); aa=$(cat /proc/self/attr/current 2>/dev/null)
echo "      uid=$(id -u) kernel=$(uname -r) Seccomp=$sec CapBnd=$cap apparmor='$aa' proc_overmounts=$(grep -c ' /proc/' /proc/mounts)"
[ "$sec" = 2 ] && ok "seccomp filter is active" || bad "seccomp filter is not active (Seccomp=$sec)" "over-grant: the session runs with seccomp=unconfined"
[ "$cap" = 00000000a88465fb ] && ok "capability set unchanged" || bad "capability set changed ($cap)" "over-grant: capabilities were added (expected 00000000a88465fb)"
case "$aa" in
  "$EXPECT_PROFILE (enforce)") ok "AppArmor profile is $EXPECT_PROFILE" ;;
  "docker-default (enforce)")  bad "AppArmor profile is still docker-default" "the new profile was not attached to this session" ;;
  unconfined)                  bad "AppArmor is unconfined" "over-grant: apparmor=unconfined was used instead of the $EXPECT_PROFILE profile" ;;
  *)                           bad "unexpected AppArmor label '$aa'" "check which profile the session got" ;;
esac

echo "== 1. seccomp: allowed and still-blocked calls"
python3 - <<'PY'
import ctypes, errno, os
l = ctypes.CDLL(None, use_errno=True)
def call(nr, *a):
    r = l.syscall(nr, *a); return "ok" if r >= 0 else errno.errorcode.get(ctypes.get_errno(), "?")
exp = [("unshare(0)", 272, (0,), "ok", "seccomp rule for unshare is missing"),
       ("setns(-1,0)", 308, (-1, 0), "EBADF", "seccomp rule for setns is missing"),
       ("clone3", 435, (None, 0), "ENOSYS", "clone3 must keep returning ENOSYS as in the default profile"),
       ("keyctl", 250, (0, -1, 0), "EPERM", "over-grant: keyctl is allowed (profile wider than requested)"),
       ("io_uring_setup", 425, (0, None), "EPERM", "over-grant: io_uring is allowed (profile wider than requested)")]
bad = 0
for name, nr, args, want, cause in exp:
    got = call(nr, *args)
    print(("  PASS  " if got == want else "  FAIL  ") + f"{name} -> {got} (expected {want})" + ("" if got == want else f"\n        cause: {cause}"))
    bad += got != want
# raw clone with namespace flags: NEWUSER must work, NEWUSER|NEWNET must be denied
for name, flags, want in (("clone(CLONE_NEWUSER)", 0x10000011, True), ("clone(CLONE_NEWUSER|CLONE_NEWNET)", 0x50000011, False)):
    r = l.syscall(56, flags, 0, 0, 0, 0)
    if r == 0: os._exit(0)
    e = ctypes.get_errno()
    if r > 0: os.waitpid(r, 0)
    good = (r > 0) == want
    print(("  PASS  " if good else "  FAIL  ") + f"{name} -> " + ("created" if r > 0 else "denied " + errno.errorcode.get(e, "?")) +
          ("" if good else "\n        cause: " + ("clone with namespace flags is not allowed" if want else "the CLONE_NEWNET argument condition is missing (name-only rule was used)")))
    bad += not good
raise SystemExit(1 if bad else 0)
PY
[ $? -eq 0 ] || n_fail=$((n_fail+1))

echo "== 2. user namespace"
out=$(unshare -Ur cat /proc/self/uid_map 2>&1); rc=$?
if [ $rc -eq 0 ]; then ok "user namespace created ($(echo $out))"
elif echo "$out" | grep -qi "permission denied"; then bad "unshare -Ur: $out" "AppArmor mediates user namespaces: the profile needs the 'userns,' rule"
else bad "unshare -Ur: $out" "seccomp still blocks unshare"; fi
out=$(unshare -Urn true 2>&1) && bad "a new network namespace could be created" "the CLONE_NEWNET condition on unshare is missing" || ok "new network namespace is refused ($out)"

echo "== 3. mount inside the user namespace"
out=$(unshare -Urm sh -c 'mount -t tmpfs none /mnt && echo mounted' 2>&1); rc=$?
if [ $rc -eq 0 ]; then ok "tmpfs mounted in a user+mount namespace"
elif echo "$out" | grep -qi "permission denied"; then bad "mount: $out" "AppArmor: 'deny mount,' is still in the profile (a deny wins over 'mount,')"
else bad "mount: $out" "seccomp still blocks mount"; fi

echo "== 4. pivot_root (not fatal: Apptainer falls back to chroot)"
out=$(unshare -Urm sh -c 'd=$(mktemp -d); mount -t tmpfs t $d; mkdir $d/o; cd $d; pivot_root . o && echo pivoted' 2>&1) && ok "pivot_root works" || note "pivot_root refused ($out) - AppArmor 'pivot_root,' rule or seccomp rule missing"

echo "== 5. signals and /proc between processes of this session (AppArmor peer rules)"
sleep 60 & p=$!
if readlink /proc/$p/ns/mnt >/dev/null 2>&1 && kill $p 2>/dev/null; then ok "can inspect and signal a sibling process"
else bad "cannot inspect or signal a sibling process" "AppArmor: signal/ptrace peer name does not match the profile name"; kill -9 $p 2>/dev/null; fi

echo "== 6. /dev/fuse"
if [ ! -c /dev/fuse ]; then bad "/dev/fuse is not a character device" "the device node was not added to the session"
else out=$(python3 -c 'import os; os.close(os.open("/dev/fuse", os.O_RDWR)); print("opened")' 2>&1) && ok "/dev/fuse opens read-write" || bad "/dev/fuse exists but cannot be opened ($(echo "$out" | tail -1))" "the device cgroup permission (rwm) is missing"; fi

command -v apptainer >/dev/null || { echo "== apptainer is not installed: steps 7+ skipped"; echo "RESULT: $n_fail failed step(s) in 0-6"; exit $((n_fail > 0)); }
echo "== 7. apptainer exec (SIF mounted through FUSE)"
out=$(apptainer exec "$ALP" true 2>&1) && ok "apptainer exec" || bad "apptainer exec: $(echo "$out" | tail -1)" "FUSE mount of the SIF (AppArmor mount rule for fuse, or squashfuse)"
echo "== 8. writable overlay"
out=$(apptainer exec --writable-tmpfs "$ALP" touch /x 2>&1) && ok "apptainer exec --writable-tmpfs" || bad "--writable-tmpfs: $(echo "$out" | tail -1)" "fuse-overlayfs path"
echo "== 9. --pid with the default Apptainer configuration (expected to fail while /proc is masked)"
out=$(apptainer exec --pid "$ALP" true 2>&1) && note "--pid works here: a fresh procfs can be mounted in this session" || note "--pid fails as expected ($(echo "$out" | grep -o "can't mount proc[^:]*" | head -1)); our own Apptainer configuration handles this"
[ -n "$SWE" ] && [ -f "$SWE" ] || { echo "== no SWE image given: steps 10+ are run separately (see t2.sh: swe_real, harbor, inst_fr)"; echo "RESULT: $n_fail failed step(s)"; exit $((n_fail > 0)); }
echo "RESULT: $n_fail failed step(s)"; exit $((n_fail > 0))
