"""Atomic no-overwrite publication of one flushed regular same-directory stage.

Hardlinks remain the first path. On Linux only, unsupported-link errno permits
the kernel's renameat2(RENAME_NOREPLACE). There is no replace, check-then-rename,
copy or syscall-number fallback. A refused kernel operation preserves its stage. Kernel
rename success consumes that new pending stage; no existing artifact is removed.
"""
from __future__ import annotations
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import sys

SCHEMA = "neural_forge.qimg_atomic_publication.v1"
VERSION = "link-or-linux-renameat2-noreplace.v1"
RENAME_NOREPLACE = 1
UNSUPPORTED_LINK_ERRNOS = frozenset({errno.EPERM,errno.EOPNOTSUPP,errno.ENOTSUP,errno.ENOSYS})


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def policy():
    return {"schema":SCHEMA,"version":VERSION,"implementation_sha256":implementation_sha256(),
        "first_method":"hardlink","linux_fallback":"libc.renameat2(RENAME_NOREPLACE)",
        "fallback_link_errnos":sorted(UNSUPPORTED_LINK_ERRNOS),"rename_flags":RENAME_NOREPLACE,
        "EPERM_scope":"actual Lexar exfat unsupported-link result; permissions are still kernel enforced",
        "atomic_scope":"complete-file visibility and no overwrite; not a power-loss durability certificate"}


def _rename_noreplace(source,destination):
    if sys.platform!="linux":raise OSError(errno.ENOSYS,"renameat2 fallback is Linux only")
    libc=ctypes.CDLL(None,use_errno=True)
    function=getattr(libc,"renameat2",None)
    if function is None:raise OSError(errno.ENOSYS,"libc exposes no renameat2; publication fails closed")
    function.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
    function.restype=ctypes.c_int
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC|os.O_NOFOLLOW
    descriptor=os.open(source.parent,flags)
    try:
        ctypes.set_errno(0)
        result=function(descriptor,os.fsencode(source.name),descriptor,os.fsencode(destination.name),RENAME_NOREPLACE)
        if result!=0:
            code=ctypes.get_errno()
            raise OSError(code or errno.EIO,os.strerror(code or errno.EIO),str(destination))
    finally:os.close(descriptor)


def publish_noreplace(stage,destination):
    """Publish already flushed stage exactly once, never overwrite destination.

    Only new producer-created regular stages in the same canonical directory are
    supported. The successful source inode is checked at the destination. Both
    syscall paths enforce exclusion atomically; target existence is never used
    as a substitute for exclusion. EPERM is an observed exfat link limitation,
    not permission elevation; rename still uses existing kernel permissions.
    """
    source=Path(stage).absolute();target=Path(destination).absolute()
    if source==target or source.parent!=target.parent or source.parent!=source.parent.resolve():
        raise ValueError("distinct canonical same-directory stage/destination required")
    initial=source.lstat()
    if not stat.S_ISREG(initial.st_mode):raise ValueError("regular nonsymlink pending stage required")
    fd=os.open(source,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0))
    link_error=None
    try:
        before=os.fstat(fd)
        if (initial.st_dev,initial.st_ino,initial.st_size)!=(before.st_dev,before.st_ino,before.st_size):
            raise ValueError("pending stage changed before publication")
        try:
            os.link(source,target,follow_symlinks=False)
            method="hardlink";retained=True
        except OSError as exc:
            if sys.platform!="linux"or exc.errno not in UNSUPPORTED_LINK_ERRNOS:raise
            link_error={"errno":exc.errno,"name":errno.errorcode.get(exc.errno,"UNKNOWN")}
            _rename_noreplace(source,target)
            method="renameat2_noreplace";retained=False
        after=target.lstat()
        if (not stat.S_ISREG(after.st_mode)or (before.st_dev,before.st_ino,before.st_size)!=(after.st_dev,after.st_ino,after.st_size)):
            raise ValueError("published inode identity differs from pending stage; no acceptance")
        if retained:
            current=source.lstat()
            if (current.st_dev,current.st_ino,current.st_size)!=(before.st_dev,before.st_ino,before.st_size):
                raise ValueError("pending stage identity changed during publication")
        elif source.exists()or source.is_symlink():
            raise ValueError("rename consumed-stage identity changed during publication")
    finally:os.close(fd)
    return {"schema":SCHEMA,"version":VERSION,"implementation_sha256":implementation_sha256(),
        "method":method,"hardlink_error":link_error,"source_retained":retained,"published":True}
