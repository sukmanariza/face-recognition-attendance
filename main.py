"""Command line entry point.

    python main.py                 launch the desktop application
    python main.py recognize       live preview window (no Qt)
    python main.py register        enroll a new face from the camera
    python main.py list            list registered identities
    python main.py deactivate 003  stop matching this identity, keep the history
    python main.py delete 003      delete an identity and its attendance
    python main.py attendance      today's presence list
    python main.py attendance --export --start 2026-11-01 --end 2026-11-30
    python main.py backup          consistent copy of the database
    python main.py audit           recent changes made through the app
    python main.py status          environment, model and database summary
    python main.py benchmark       measure performance and write a JSON report
    python main.py settings        show the effective configuration

Every command loads `config.yaml`, then the hardware performance tier, then
`settings.json`, so the CLI and the window always behave identically. A database
key protected by a passphrase is unlocked with `--passphrase` or the
FACEID_PASSPHRASE environment variable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import cv2

from config import ConfigError, load_config
from src.application import Application
from src.database import (
    DatabaseError,
    DuplicateEmployeeError,
    EmployeeNotFoundError,
)
from src.face_registration import RegistrationError
from src.pipeline import CameraSource, Mode, RecognitionPipeline
from src.ui import (
    render_idle,
    render_recognition,
    render_registration,
    resize_for_display,
)

LOGGER = logging.getLogger("faceid")
_QUIT_KEYS = {ord("q"), 27}  # q or Esc
_WINDOW_NAME = "Face ID"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_application(args: argparse.Namespace) -> Application:
    config_path = Path(args.config).expanduser() if args.config else None
    config = load_config(config_path)
    configure_logging(config.log_level)
    application = Application(config, passphrase=getattr(args, "passphrase", None))
    if config.log_level != application.config.log_level:
        configure_logging(application.config.log_level)
    return application


def command_gui(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        from src.gui import run_app

        return run_app(application)
    except ImportError as exc:
        print(f"The desktop interface needs PySide6: {exc}", file=sys.stderr)
        print("Run `python install.py`, or use `python main.py recognize`.", file=sys.stderr)
        return 2
    finally:
        application.close()


def command_status(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        for line in application.describe_environment():
            print(line)
        database = application.database
        records = database.list_employees()
        print(f"Identities        {len(records)} ({database.count_employees()} rows)")
        for record in records:
            print(f"  {record.summary}")
        if database.model_signature:
            print(f"Embeddings from   {database.model_signature}")
        cameras = application.probe_cameras()
        print(f"Cameras           {cameras if cameras else 'none detected'}")
    except DatabaseError as exc:
        print(f"Database problem: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()
    return 0


def command_settings(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        print(f"Configuration    {application.config.source_path}")
        print(f"Settings file    {application.settings.path}")
        print(f"Performance tier {application.tier}")
        for name, values in application.config.to_mapping().items():
            print(f"\n[{name}]")
            for key, value in values.items():
                print(f"  {key:<28} {value}")
        if not application.settings.path.is_file():
            print("\nNo settings.json yet; values above come from config.yaml and the tier.")
    finally:
        application.close()
    return 0


def command_list(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        records = application.database.list_employees()
    except DatabaseError as exc:
        print(f"Database problem: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()

    if not records:
        print("No identities registered. Run `python main.py register`.")
        return 0
    print(f"{len(records)} identit{'y' if len(records) == 1 else 'ies'}:")
    for record in records:
        print(f"  {record.summary}")
    return 0


def command_delete(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        deleted = application.database.delete_employee(args.employee_id)
    except (DatabaseError, EmployeeNotFoundError) as exc:
        print(f"Could not delete {args.employee_id}: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()
    print(f"Deleted {args.employee_id}." if deleted else f"{args.employee_id} was not found.")
    return 0 if deleted else 1


def command_deactivate(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        record = application.database.set_employee_active(args.employee_id, not args.undo)
    except (DatabaseError, EmployeeNotFoundError) as exc:
        print(f"Could not update {args.employee_id}: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()
    state = "active" if record.active else "inactive"
    print(f"{record.employee_id} ({record.full_name}) is now {state}.")
    return 0


def command_attendance(args: argparse.Namespace) -> int:
    application = _load_application(args)
    service = application.attendance
    try:
        if args.export:
            start = date.fromisoformat(args.start) if args.start else None
            end = date.fromisoformat(args.end) if args.end else None
            stamp = start or date.today()
            target = application.config.attendance_dir / f"attendance-{stamp:%Y-%m}.csv"
            path = service.export_csv(target, start, end)
            print(f"Exported {path}")
            return 0
        if args.delete is not None:
            deleted = service.delete(args.delete)
            print("Deleted." if deleted else "That row no longer exists.")
            return 0 if deleted else 1
        work_date = args.date or service.today()
        entries = service.day(work_date)
    except (DatabaseError, ValueError, OSError) as exc:
        print(f"Attendance error: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()

    if not entries:
        print(f"Nobody recorded on {work_date}.")
        return 0
    late = sum(1 for entry in entries if entry.status == "late")
    print(f"{len(entries)} present on {work_date} ({late} late):")
    for entry in entries:
        print(f"  {entry.summary}")
    return 0


def command_backup(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = application.config.database_path.parent / "backups" / f"faces-{stamp}.db"
        path = application.database.backup_to(target)
        key = application.config.key_path
    except (DatabaseError, OSError) as exc:
        print(f"Backup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()
    print(f"Database backed up to {path}")
    if application.config.database.encrypt:
        print(f"Copy the key as well, or the backup cannot be read: {key}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    application = _load_application(args)
    try:
        entries = application.database.audit_entries(args.limit)
    except DatabaseError as exc:
        print(f"Could not read the audit trail: {exc}", file=sys.stderr)
        return 1
    finally:
        application.close()
    if not entries:
        print("The audit trail is empty.")
        return 0
    for at, action, detail in entries:
        print(f"{at}  {action:<20} {detail}")
    return 0


def command_recognize(args: argparse.Namespace) -> int:
    application = _load_application(args)
    config = application.config
    pipeline: RecognitionPipeline | None = None
    try:
        session = application.create_recognition_session()
        camera = application.create_camera()
        pipeline = RecognitionPipeline(config, CameraSource(camera))
        pipeline.set_recognition_session(session)
        pipeline.set_mode(Mode.RECOGNIZE)
        camera.open()
        pipeline.start()
        print("Press q in the video window to stop.")
        deadline = time.monotonic() + args.seconds if args.seconds else None
        while True:
            output = pipeline.wait_output(timeout=0.2)
            if output is None:
                continue
            image = (
                render_recognition(output.frame.image, output.recognition, config.ui)
                if output.recognition is not None
                else render_idle(output.frame.image, config.ui)
            )
            cv2.imshow(_WINDOW_NAME, image)
            pipeline.note_displayed()
            if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                break
            if deadline is not None and time.monotonic() > deadline:
                break
    except Exception as exc:  # noqa: BLE001 - CLI must print, never traceback-dump
        LOGGER.error("%s", exc)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()
        cv2.destroyAllWindows()
        application.close()
    return 0


def command_register(args: argparse.Namespace) -> int:
    application = _load_application(args)
    config = application.config
    pipeline: RecognitionPipeline | None = None
    try:
        database = application.database
        employee_id = args.employee_id or database.next_employee_id()
        existing = database.get_employee(employee_id)
        if existing is not None and not args.replace:
            print(
                f"ID {employee_id} is already registered. Use another ID, or "
                f"`register --id {employee_id} --replace` to record the face again.",
                file=sys.stderr,
            )
            return 1
        default_name = existing.full_name if existing is not None else ""
        full_name = args.name or default_name
        if not full_name:
            full_name = input(f"Full name for {employee_id}: ").strip()
        if not full_name:
            print("A name is required.", file=sys.stderr)
            return 1

        session = application.create_registration_session()
        camera = application.create_camera()
        pipeline = RecognitionPipeline(config, CameraSource(camera))
        pipeline.begin_registration(session)
        camera.open()
        pipeline.start()
        print(f"Registering {employee_id} - {full_name}. Follow the on-screen steps, q to abort.")

        last_message = ""
        while True:
            output = pipeline.wait_output(timeout=0.2)
            if output is None:
                continue
            attempt = output.attempt
            if attempt is not None:
                image = render_registration(output.frame.image, attempt, config.ui)
                cv2.imshow(_WINDOW_NAME, image)
                if attempt.message != last_message:
                    print(f"  [{attempt.status}] {attempt.message}")
                    last_message = attempt.message
                if attempt.is_complete:
                    break
            else:
                image = resize_for_display(output.frame.image, config.ui.display_max_width)
                cv2.imshow(_WINDOW_NAME, image)
            if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                print("Cancelled; nothing was saved.")
                return 1

        samples = session.build_samples()
        registrar = application.create_registrar()
        result = (
            registrar.save_update(
                employee_id, samples, session.min_consistency(), full_name=full_name
            )
            if existing is not None
            else registrar.save_new(employee_id, full_name, samples, session.min_consistency())
        )
        application.invalidate_gallery()
        print(result.describe())
    except (RegistrationError, DatabaseError, DuplicateEmployeeError) as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI must print, never traceback-dump
        LOGGER.error("%s", exc)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()
        cv2.destroyAllWindows()
        application.close()
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from src.benchmark import run_benchmark

    application = _load_application(args)
    try:
        print("Benchmarking the live pipeline. Keep the camera pointed at a face.")
        report = run_benchmark(
            application,
            duration=args.duration,
            warmup=args.warmup,
        )
        print(report.describe())
        target = application.config.benchmark.target_display_fps
        print(report.verdict(target))
        path = report.write(application.config.benchmark_dir)
        print(f"Report written to {path}")
    except Exception as exc:  # noqa: BLE001 - CLI must print, never traceback-dump
        LOGGER.error("Benchmark failed: %s", exc)
        return 1
    finally:
        application.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-id",
        description="Local face recognition with InsightFace, ONNX Runtime, and SQLite.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--passphrase",
        default=None,
        help="passphrase for an encrypted database (or set FACEID_PASSPHRASE)",
    )
    parser.add_argument("--version", action="version", version="Face ID 1.0")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gui", help="launch the desktop application (default)")
    subparsers.add_parser("status", help="show environment, model, and database summary")
    subparsers.add_parser("settings", help="show the effective configuration")
    subparsers.add_parser("list", help="list registered identities")

    delete = subparsers.add_parser("delete", help="delete an identity and its attendance")
    delete.add_argument("employee_id", help="ID to delete, for example 003")

    deactivate = subparsers.add_parser(
        "deactivate", help="stop matching an identity but keep its history"
    )
    deactivate.add_argument("employee_id", help="ID to deactivate, for example 003")
    deactivate.add_argument("--undo", action="store_true", help="reactivate instead")

    attendance = subparsers.add_parser("attendance", help="show or export attendance")
    attendance.add_argument("--date", default=None, help="day to list, YYYY-MM-DD")
    attendance.add_argument("--export", action="store_true", help="write a CSV instead")
    attendance.add_argument("--start", default=None, help="export from this day")
    attendance.add_argument("--end", default=None, help="export up to this day")
    attendance.add_argument("--delete", type=int, default=None, help="delete one row by id")

    subparsers.add_parser("backup", help="write a consistent copy of the database")

    audit = subparsers.add_parser("audit", help="recent changes made through the app")
    audit.add_argument("--limit", type=int, default=25)
    recognize = subparsers.add_parser("recognize", help="live recognition preview window")
    recognize.add_argument(
        "--seconds", type=float, default=0.0, help="stop after N seconds (0 = until q)"
    )

    register = subparsers.add_parser("register", help="enroll a new face")
    register.add_argument("--id", dest="employee_id", default=None, help="ID (default: next free)")
    register.add_argument("--name", default=None, help="full name")
    register.add_argument(
        "--replace",
        action="store_true",
        help="record the face again for an ID that already exists",
    )

    benchmark = subparsers.add_parser("benchmark", help="measure performance")
    benchmark.add_argument("--duration", type=float, default=None, help="measured seconds")
    benchmark.add_argument("--warmup", type=float, default=None, help="warmup seconds")

    return parser


_HANDLERS = {
    None: command_gui,
    "gui": command_gui,
    "status": command_status,
    "settings": command_settings,
    "list": command_list,
    "delete": command_delete,
    "deactivate": command_deactivate,
    "attendance": command_attendance,
    "backup": command_backup,
    "audit": command_audit,
    "recognize": command_recognize,
    "register": command_register,
    "benchmark": command_benchmark,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command or None)
    if handler is None:  # pragma: no cover - argparse rejects unknown commands
        parser.print_help()
        return 2
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
