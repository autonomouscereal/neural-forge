"""Real publication races plus fail-closed API tests; no model or Torch import."""
import errno
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import c16_qimg_atomic_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name).resolve()
    def tearDown(self):self.directory.cleanup()
    def stage(self,name="pending",data=b"complete flushed bytes"):
        p=self.root/name
        with p.open("xb")as f:f.write(data);f.flush();os.fsync(f.fileno())
        return p
    def test_hardlink_retains_exact_stage_and_bytes(self):
        stage=self.stage();target=self.root/"target"
        result=publication.publish_noreplace(stage,target)
        self.assertEqual(result["method"],"hardlink")
        self.assertTrue(result["source_retained"])
        self.assertEqual(stage.read_bytes(),target.read_bytes())
        self.assertEqual(result["implementation_sha256"],publication.implementation_sha256())
    def test_existing_file_never_overwritten(self):
        stage=self.stage();target=self.stage("target",b"existing immutable")
        with self.assertRaises(OSError)as caught:publication.publish_noreplace(stage,target)
        self.assertEqual(caught.exception.errno,errno.EEXIST)
        self.assertEqual(target.read_bytes(),b"existing immutable")
        self.assertTrue(stage.exists())
    def test_other_link_errors_do_not_call_fallback(self):
        for code in (errno.EACCES,errno.EEXIST,errno.ENOSPC,errno.EXDEV,errno.EIO,errno.EMLINK):
            with self.subTest(code=code),patch.object(publication.os,"link",side_effect=OSError(code,"forced")),patch.object(publication,"_rename_noreplace")as rename:
                stage=self.stage(str(code))
                with self.assertRaises(OSError)as caught:publication.publish_noreplace(stage,self.root/(str(code)+"out"))
                self.assertEqual(caught.exception.errno,code);rename.assert_not_called()
                self.assertTrue(stage.exists())
    def test_nonlinux_unsupported_link_fails_closed(self):
        stage=self.stage()
        with patch.object(publication.sys,"platform","win32"),patch.object(publication.os,"link",side_effect=OSError(errno.EPERM,"unsupported")),patch.object(publication,"_rename_noreplace")as rename:
            with self.assertRaises(OSError):publication.publish_noreplace(stage,self.root/"out")
            rename.assert_not_called()
        self.assertTrue(stage.exists());self.assertFalse((self.root/"out").exists())
    def test_unsupported_rename_fails_closed(self):
        for code in (errno.ENOSYS,errno.EOPNOTSUPP,errno.EINVAL,errno.EACCES,errno.ENOSPC):
            with self.subTest(code=code),patch.object(publication.sys,"platform","linux"),patch.object(publication.os,"link",side_effect=OSError(errno.EPERM,"unsupported")),patch.object(publication,"_rename_noreplace",side_effect=OSError(code,"kernel refused")):
                stage=self.stage(str(code));target=self.root/(str(code)+"out")
                with self.assertRaises(OSError)as caught:publication.publish_noreplace(stage,target)
                self.assertEqual(caught.exception.errno,code)
                self.assertTrue(stage.exists());self.assertFalse(target.exists())
    def test_no_libc_symbol_fails_closed(self):
        with patch.object(publication.sys,"platform","linux"),patch.object(publication.ctypes,"CDLL",return_value=object()):
            with self.assertRaises(OSError)as caught:publication._rename_noreplace(self.root/"x",self.root/"y")
        self.assertEqual(caught.exception.errno,errno.ENOSYS)
    def test_distinct_same_directory_regular_stage_required(self):
        stage=self.stage();other=self.root/"different";other.mkdir()
        for target in (stage,other/"out"):
            with self.assertRaises(ValueError):publication.publish_noreplace(stage,target)
        with self.assertRaises(ValueError):publication.publish_noreplace(other,self.root/"out")
        self.assertFalse((self.root/"out").exists())
    def race(self,forced_fallback):
        count=16;barrier=threading.Barrier(count);lock=threading.Lock();answers=[]
        stages=[self.stage(f"pending{i}",f"winner complete {i}:".encode()+bytes([i])*32768)for i in range(count)]
        payloads=[p.read_bytes()for p in stages];target=self.root/"raceout"
        def call(i):
            barrier.wait(timeout=10)
            try:answer=(i,publication.publish_noreplace(stages[i],target),None)
            except OSError as exc:answer=(i,None,exc.errno)
            with lock:answers.append(answer)
        ctx=patch.object(publication.os,"link",side_effect=OSError(errno.EPERM,"exfat fixture"))if forced_fallback else patch.object(publication.os,"link",wraps=os.link)
        with ctx:
            threads=[threading.Thread(target=call,args=(i,))for i in range(count)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(timeout=15);self.assertFalse(thread.is_alive())
        self.assertEqual(len(answers),count)
        winners=[a for a in answers if a[1]is not None]
        self.assertEqual(len(winners),1)
        winner=winners[0][0];self.assertEqual(target.read_bytes(),payloads[winner])
        self.assertEqual(sum(a[2]==errno.EEXIST for a in answers),count-1)
        for i,p in enumerate(stages):
            if i==winner and forced_fallback:self.assertFalse(p.exists())
            else:self.assertEqual(p.read_bytes(),payloads[i])
    def test_actual_hardlink_concurrency(self):self.race(False)
    @unittest.skipUnless(sys.platform=="linux","actual Linux libc/kernel required")
    def test_actual_renameat2_concurrency(self):self.race(True)
    @unittest.skipUnless(sys.platform=="linux","actual Linux libc/kernel required")
    def test_actual_rename_consumes_stage_and_refuses_existing_target(self):
        stage=self.stage();target=self.root/"out"
        with patch.object(publication.os,"link",side_effect=OSError(errno.EPERM,"exfat fixture")):
            result=publication.publish_noreplace(stage,target)
            self.assertEqual(result["method"],"renameat2_noreplace");self.assertFalse(stage.exists())
            conflict=self.stage("conflict",b"loser")
            with self.assertRaises(OSError)as caught:publication.publish_noreplace(conflict,target)
            self.assertEqual(caught.exception.errno,errno.EEXIST)
            self.assertEqual(conflict.read_bytes(),b"loser");self.assertEqual(target.read_bytes(),b"complete flushed bytes")
    @unittest.skipUnless(sys.platform=="linux","actual Linux filesystem symlink required")
    def test_symlink_destination_is_never_followed_or_overwritten(self):
        target=self.root/"out";target.symlink_to(self.root/"absent")
        for i,forced in enumerate((False,True)):
            stage=self.stage(str(i))
            ctx=patch.object(publication.os,"link",side_effect=OSError(errno.EPERM,"exfat fixture"))if forced else patch.object(publication.os,"link",wraps=os.link)
            with ctx,self.assertRaises(OSError)as caught:publication.publish_noreplace(stage,target)
            self.assertEqual(caught.exception.errno,errno.EEXIST)
            self.assertTrue(target.is_symlink());self.assertTrue(stage.exists());self.assertFalse((self.root/"absent").exists())
    @unittest.skipUnless(sys.platform=="linux","actual Linux filesystem symlink required")
    def test_stage_and_parent_symlinks_are_refused(self):
        stage=self.stage();alias=self.root/"alias";alias.symlink_to(stage)
        with self.assertRaises(ValueError):publication.publish_noreplace(alias,self.root/"out")
        real=self.root/"real";real.mkdir();(real/"pending").write_bytes(b"x")
        parent=self.root/"parent";parent.symlink_to(real,target_is_directory=True)
        with self.assertRaises(ValueError):publication.publish_noreplace(parent/"pending",parent/"out")
        self.assertFalse((real/"out").exists())


if __name__=="__main__":unittest.main()
