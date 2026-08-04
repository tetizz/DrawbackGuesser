import { AsyncLocalStorage } from "node:async_hooks";
import { execFile, spawn, type ChildProcess } from "node:child_process";
import { lstat, realpath } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, sep } from "node:path";

const DEFAULT_COMMAND_TIMEOUT_MS = 10 * 60 * 1000;
const DEFAULT_MAX_OUTPUT_BYTES = 16 * 1024 * 1024;
const DEFAULT_CLEANUP_SETTLEMENT_MS = 10_000;
const TASKKILL_TIMEOUT_MS = 10_000;
const WINDOWS_SYSTEM_ROOT_ALIAS = String.raw`\\?\GLOBALROOT\SystemRoot`;

const signalContext = new AsyncLocalStorage<AbortSignal>();

export interface Schema9CommandResult {
  readonly stdout: string;
  readonly stderr: string;
}

export interface Schema9CommandOptions {
  readonly cwd: string;
  readonly environment: NodeJS.ProcessEnv;
  readonly signal?: AbortSignal;
  readonly timeoutMilliseconds?: number;
  readonly maxOutputBytes?: number;
  readonly cleanupSettlementMilliseconds?: number;
  readonly description?: string;
}

export class Schema9CommandFailure extends Error {
  public constructor(
    message: string,
    public readonly code: number | null,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "Schema9CommandFailure";
  }
}

export type Schema9TerminationSignal = "SIGINT" | "SIGTERM";

export class Schema9TerminationError extends Error {
  public constructor(
    public readonly signal: Schema9TerminationSignal,
    public readonly exitCode: number,
  ) {
    super(`Schema-9 ledger operation was interrupted by ${signal}.`);
    this.name = "Schema9TerminationError";
  }
}

export interface InstalledSchema9TerminationSignal {
  readonly signal: AbortSignal;
  abort(reason: Error): void;
  dispose(): void;
}

interface TerminationSignalSource {
  on(signal: Schema9TerminationSignal, listener: () => void): void;
  removeListener(
    signal: Schema9TerminationSignal,
    listener: () => void,
  ): void;
}

interface AuthenticatedSystemTool {
  readonly path: string;
  readonly dev: bigint;
  readonly ino: bigint;
  readonly size: bigint;
  readonly mtimeNs: bigint;
  readonly ctimeNs: bigint;
  readonly environment: NodeJS.ProcessEnv;
}

interface Schema9CommandLaunch {
  readonly command: string;
  readonly arguments: readonly string[];
  readonly environment: NodeJS.ProcessEnv;
  readonly standardInput?: Buffer;
  readonly authenticatedLauncher?: AuthenticatedSystemTool;
}

/** Makes one cancellation signal available to every nested command. */
export function withSchema9CommandSignal<T>(
  signal: AbortSignal,
  operation: () => Promise<T>,
): Promise<T> {
  return signalContext.run(signal, operation);
}

/** Converts catchable termination signals into cooperative cleanup. */
export function installSchema9TerminationSignal(
  source: TerminationSignalSource = process,
): InstalledSchema9TerminationSignal {
  const controller = new AbortController();
  const interrupt = (): void => {
    abortOnce(controller, "SIGINT", 130);
  };
  const terminate = (): void => {
    abortOnce(controller, "SIGTERM", 143);
  };
  source.on("SIGINT", interrupt);
  source.on("SIGTERM", terminate);
  let disposed = false;
  return Object.freeze({
    signal: controller.signal,
    abort(reason: Error): void {
      if (!controller.signal.aborted) {
        controller.abort(reason);
      }
    },
    dispose(): void {
      if (disposed) {
        return;
      }
      disposed = true;
      source.removeListener("SIGINT", interrupt);
      source.removeListener("SIGTERM", terminate);
    },
  });
}

export function findSchema9TerminationError(
  value: unknown,
): Schema9TerminationError | undefined {
  const pending: unknown[] = [value];
  const seen = new Set<unknown>();
  while (pending.length > 0) {
    const current = pending.pop();
    if (current === undefined || seen.has(current)) {
      continue;
    }
    seen.add(current);
    if (current instanceof Schema9TerminationError) {
      return current;
    }
    if (current instanceof AggregateError) {
      pending.push(...current.errors as readonly unknown[]);
    }
    if (current instanceof Error && current.cause !== undefined) {
      pending.push(current.cause);
    }
  }
  return undefined;
}

/** Runs one bounded child and proves whole-tree cleanup on interruption. */
export async function runSchema9BoundedCommand(
  command: string,
  arguments_: readonly string[],
  options: Schema9CommandOptions,
): Promise<Schema9CommandResult> {
  const signal = options.signal ?? signalContext.getStore();
  throwIfAborted(signal);
  const timeoutMilliseconds = options.timeoutMilliseconds
    ?? DEFAULT_COMMAND_TIMEOUT_MS;
  const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
  const cleanupSettlementMilliseconds = options.cleanupSettlementMilliseconds
    ?? DEFAULT_CLEANUP_SETTLEMENT_MS;
  if (!Number.isSafeInteger(timeoutMilliseconds) || timeoutMilliseconds <= 0) {
    throw new RangeError("Schema-9 command timeout must be a positive integer.");
  }
  if (!Number.isSafeInteger(maxOutputBytes) || maxOutputBytes <= 0) {
    throw new RangeError("Schema-9 command output limit must be a positive integer.");
  }
  if (
    !Number.isSafeInteger(cleanupSettlementMilliseconds)
    || cleanupSettlementMilliseconds <= 0
  ) {
    throw new RangeError(
      "Schema-9 cleanup settlement limit must be a positive integer.",
    );
  }
  const description = options.description ?? "command";
  const launch = await prepareSchema9CommandLaunch(
    command,
    arguments_,
    options,
  );
  throwIfAborted(signal);

  return new Promise((accept, reject) => {
    let timedOut = false;
    let outputExceeded = false;
    let launchFailure: Error | undefined;
    let termination: Promise<void> | undefined;
    let cleanupSettlement: ReturnType<typeof setTimeout> | undefined;
    let settled = false;
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let outputBytes = 0;
    const child = spawn(launch.command, [...launch.arguments], {
      cwd: options.cwd,
      env: launch.environment,
      windowsHide: true,
      detached: process.platform !== "win32",
      stdio: [launch.standardInput === undefined ? "ignore" : "pipe", "pipe", "pipe"],
    });
    const childStdout = child.stdout;
    const childStderr = child.stderr;
    if (childStdout === null || childStderr === null) {
      const pipeFailure = new Error(
        "Schema-9 command output pipes were not created.",
      );
      void terminateSchema9ProcessTree(child).then(
        () => {
          reject(pipeFailure);
        },
        (cleanupFailure: unknown) => {
          reject(new AggregateError(
            [pipeFailure, cleanupFailure],
            "Schema-9 output-pipe and process-tree cleanup both failed.",
          ));
        },
      );
      return;
    }
    if (launch.standardInput !== undefined) {
      child.stdin?.once("error", (error) => {
        launchFailure ??= error;
        terminateTree();
      });
      child.stdin?.end(launch.standardInput);
    }
    const forceSettlement = (treeFailure: unknown): void => {
      cleanupSettlement ??= setTimeout(() => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeout);
        signal?.removeEventListener("abort", abort);
        childStdout.destroy();
        childStderr.destroy();
        child.unref();
        const primaryFailure = signal?.aborted === true
          ? (signal.reason instanceof Error
            ? signal.reason
            : new Error("Schema-9 command was interrupted."))
          : timedOut
            ? new Error(`Schema-9 ${description} exceeded its time limit.`)
            : outputExceeded
              ? new Error(
                `Schema-9 ${description} produced excessive output.`,
              )
              : launchFailure !== undefined
                ? new Schema9CommandFailure(
                  `Schema-9 ${description} failed.`,
                  null,
                  { cause: launchFailure },
                )
              : new Error(`Schema-9 ${description} cleanup was required.`);
        reject(new AggregateError(
          [primaryFailure, treeFailure],
          `Schema-9 ${description} process-tree cleanup did not settle.`,
        ));
      }, cleanupSettlementMilliseconds);
    };
    const terminateTree = (): void => {
      if (termination === undefined) {
        termination = terminateSchema9ProcessTree(child).catch(
          (treeFailure: unknown) => {
            let directFailure: unknown;
            try {
              child.kill("SIGKILL");
            } catch (error: unknown) {
              directFailure = error;
            }
            if (directFailure !== undefined) {
              throw new AggregateError(
                [treeFailure, directFailure],
                "Schema-9 process-tree and direct-child cleanup both failed.",
              );
            }
            throw treeFailure instanceof Error
              ? treeFailure
              : new Error("Schema-9 process-tree cleanup failed.", {
                cause: treeFailure,
              });
          },
        );
        void termination.then(
          () => {
            forceSettlement(new Error(
              "Schema-9 process tree reported termination before its streams settled.",
            ));
          },
          (treeFailure: unknown) => {
            forceSettlement(treeFailure);
          },
        );
      }
    };
    const abort = (): void => {
      terminateTree();
    };
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted === true) {
      abort();
    }
    const timeout = setTimeout(() => {
      timedOut = true;
      terminateTree();
    }, timeoutMilliseconds);
    const capture = (target: Buffer[], chunk: Buffer): void => {
      outputBytes += chunk.byteLength;
      if (outputBytes <= maxOutputBytes) {
        target.push(chunk);
        return;
      }
      outputExceeded = true;
      terminateTree();
    };
    childStdout.on("data", (chunk: Buffer) => {
      capture(stdout, chunk);
    });
    childStderr.on("data", (chunk: Buffer) => {
      capture(stderr, chunk);
    });
    child.once("error", (error) => {
      launchFailure = error;
    });
    child.once("close", (code, exitSignal) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      if (cleanupSettlement !== undefined) {
        clearTimeout(cleanupSettlement);
      }
      signal?.removeEventListener("abort", abort);
      void (async () => {
        let terminationFailure: unknown;
        try {
          await termination;
        } catch (error: unknown) {
          terminationFailure = error;
        }
        if (launch.authenticatedLauncher !== undefined) {
          try {
            await assertAuthenticatedSystemToolUnchanged(
              launch.authenticatedLauncher,
            );
          } catch (error: unknown) {
            terminationFailure = terminationFailure === undefined
              ? error
              : new AggregateError(
                [terminationFailure, error],
                "Schema-9 command launcher and cleanup verification both failed.",
              );
          }
        }
        if (signal?.aborted === true) {
          const interruption = signal.reason instanceof Error
            ? signal.reason
            : new Error("Schema-9 command was interrupted.");
          throw terminationFailure === undefined
            ? interruption
            : new AggregateError(
              [interruption, terminationFailure],
              `Schema-9 ${description} interruption cleanup failed.`,
            );
        }
        if (timedOut) {
          const timeoutFailure = new Error(
            `Schema-9 ${description} exceeded its time limit.`,
          );
          throw terminationFailure === undefined
            ? timeoutFailure
            : new AggregateError(
              [timeoutFailure, terminationFailure],
              `Schema-9 ${description} timed out and tree cleanup failed.`,
            );
        }
        if (outputExceeded) {
          throw preserveSchema9PrimaryWithCleanup(
            new Error(`Schema-9 ${description} produced excessive output.`),
            terminationFailure,
            `Schema-9 ${description} output-limit cleanup failed.`,
          );
        }
        const decodedStdout = Buffer.concat(stdout).toString("utf8");
        const decodedStderr = Buffer.concat(stderr).toString("utf8");
        if (launchFailure !== undefined || code !== 0 || exitSignal !== null) {
          const commandFailure = new Schema9CommandFailure(
            `Schema-9 ${description} failed.`,
            code,
            {
              cause: launchFailure ?? new Error(
                decodedStderr.trim().length === 0
                  ? `exit=${String(code)} signal=${String(exitSignal)}`
                  : decodedStderr.trim(),
              ),
            },
          );
          throw preserveSchema9PrimaryWithCleanup(
            commandFailure,
            terminationFailure,
            `Schema-9 ${description} failed and cleanup verification failed.`,
          );
        }
        if (terminationFailure !== undefined) {
          throw terminationFailure instanceof Error
            ? terminationFailure
            : new Error("Schema-9 process-tree cleanup failed.", {
              cause: terminationFailure,
            });
        }
        return Object.freeze({
          stdout: decodedStdout,
          stderr: decodedStderr,
        });
      })().then(accept, reject);
    });
  });
}

async function prepareSchema9CommandLaunch(
  command: string,
  arguments_: readonly string[],
  options: Schema9CommandOptions,
): Promise<Schema9CommandLaunch> {
  if (command.includes("\0")) {
    throw new TypeError("Schema-9 command executable contains a null byte.");
  }
  if (options.cwd.includes("\0")) {
    throw new TypeError(
      "Schema-9 command working directory contains a null byte.",
    );
  }
  for (const [index, argument] of arguments_.entries()) {
    if (argument.includes("\0")) {
      throw new TypeError(
        `Schema-9 command argument ${String(index)} contains a null byte.`,
      );
    }
  }
  if (process.platform !== "win32") {
    return Object.freeze({
      command,
      arguments: Object.freeze([...arguments_]),
      environment: options.environment,
    });
  }
  const launcher = await authenticatedWindowsPowerShell();
  const commandLine = [command, ...arguments_]
    .map(quoteWindowsCommandLineArgument)
    .join(" ");
  const targetEntries = new Map<string, readonly [string, string]>();
  for (const [name, value] of Object.entries(options.environment)) {
    if (value === undefined) {
      continue;
    }
    const normalized = name.toLocaleUpperCase("en-US");
    if (targetEntries.has(normalized)) {
      throw new TypeError(
        "Schema-9 command environment contains case-insensitive duplicates.",
      );
    }
    targetEntries.set(normalized, Object.freeze([name, value]));
  }
  for (const name of ["SystemRoot", "WINDIR", "ComSpec"] as const) {
    const value = launcher.environment[name];
    if (value === undefined) {
      throw new TypeError("Schema-9 supervisor environment is incomplete.");
    }
    targetEntries.set(
      name.toLocaleUpperCase("en-US"),
      Object.freeze([name, value]),
    );
  }
  const targetEnvironment = Object.fromEntries(targetEntries.values());
  const environmentBlock = serializeWindowsEnvironmentBlock(targetEnvironment);
  const script = windowsJobSupervisorScript(commandLine, options.cwd);
  return Object.freeze({
    command: launcher.path,
    arguments: Object.freeze([
      "-NoLogo",
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-EncodedCommand",
      Buffer.from(script, "utf16le").toString("base64"),
    ]),
    environment: launcher.environment,
    standardInput: Buffer.from(environmentBlock.toString("base64"), "ascii"),
    authenticatedLauncher: launcher,
  });
}

function serializeWindowsEnvironmentBlock(
  environment: NodeJS.ProcessEnv,
): Buffer {
  const entries: Array<readonly [string, string, string]> = [];
  const names = new Set<string>();
  for (const [name, value] of Object.entries(environment)) {
    if (value === undefined) {
      continue;
    }
    if (
      name.length === 0
      || name.includes("\0")
      || name.includes("=")
      || value.includes("\0")
    ) {
      throw new TypeError("Schema-9 command environment is not representable.");
    }
    const normalized = name.toLocaleUpperCase("en-US");
    if (names.has(normalized)) {
      throw new TypeError(
        "Schema-9 command environment contains case-insensitive duplicates.",
      );
    }
    names.add(normalized);
    entries.push(Object.freeze([normalized, name, value]));
  }
  entries.sort((left, right) => {
    const byName = left[0].localeCompare(right[0], "en-US");
    return byName === 0 ? left[1].localeCompare(right[1], "en-US") : byName;
  });
  const block = `${entries.map((entry) => `${entry[1]}=${entry[2]}`).join("\0")}\0\0`;
  return Buffer.from(block, "utf16le");
}

function quoteWindowsCommandLineArgument(value: string): string {
  if (value.length > 0 && !/[\s"]/u.test(value)) {
    return value;
  }
  let encoded = '"';
  let backslashes = 0;
  for (const character of value) {
    if (character === "\\") {
      backslashes += 1;
      continue;
    }
    if (character === '"') {
      encoded += "\\".repeat(backslashes * 2 + 1) + '"';
      backslashes = 0;
      continue;
    }
    encoded += "\\".repeat(backslashes) + character;
    backslashes = 0;
  }
  return `${encoded}${"\\".repeat(backslashes * 2)}"`;
}

function windowsJobSupervisorScript(
  commandLine: string,
  workingDirectory: string,
): string {
  const encodedCommand = Buffer.from(commandLine, "utf8").toString("base64");
  const encodedDirectory = Buffer.from(workingDirectory, "utf8")
    .toString("base64");
  return String.raw`
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PSModulePath = $null
$forbiddenHostNames = @(
  "COR_ENABLE_PROFILING",
  "COR_PROFILER",
  "CORECLR_ENABLE_PROFILING",
  "CORECLR_PROFILER",
  "PSModulePath"
)
$hostEnvironment = [Environment]::GetEnvironmentVariables()
foreach ($name in $forbiddenHostNames) {
  if ($hostEnvironment.Contains($name)) {
    throw "Schema-9 supervisor inherited a forbidden host variable."
  }
}
foreach ($name in $hostEnvironment.Keys) {
  if ([string]$name -like "COMPlus_*") {
    throw "Schema-9 supervisor inherited a forbidden COMPlus host variable."
  }
}
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class Schema9JobSupervisor {
  private const UInt32 CREATE_SUSPENDED = 0x00000004;
  private const UInt32 CREATE_UNICODE_ENVIRONMENT = 0x00000400;
  private const UInt32 INFINITE = 0xFFFFFFFF;
  private const UInt32 JOB_DRAIN_TIMEOUT_MILLISECONDS = 10000;
  private const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
  private const UInt32 WAIT_OBJECT_0 = 0x00000000;
  private const Int32 JobObjectExtendedLimitInformation = 9;

  [StructLayout(LayoutKind.Sequential)]
  private struct IO_COUNTERS {
    public UInt64 ReadOperationCount;
    public UInt64 WriteOperationCount;
    public UInt64 OtherOperationCount;
    public UInt64 ReadTransferCount;
    public UInt64 WriteTransferCount;
    public UInt64 OtherTransferCount;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    public Int64 PerProcessUserTimeLimit;
    public Int64 PerJobUserTimeLimit;
    public UInt32 LimitFlags;
    public UIntPtr MinimumWorkingSetSize;
    public UIntPtr MaximumWorkingSetSize;
    public UInt32 ActiveProcessLimit;
    public UIntPtr Affinity;
    public UInt32 PriorityClass;
    public UInt32 SchedulingClass;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
    public IO_COUNTERS IoInfo;
    public UIntPtr ProcessMemoryLimit;
    public UIntPtr JobMemoryLimit;
    public UIntPtr PeakProcessMemoryUsed;
    public UIntPtr PeakJobMemoryUsed;
  }

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  private struct STARTUPINFO {
    public UInt32 cb;
    public string lpReserved;
    public string lpDesktop;
    public string lpTitle;
    public UInt32 dwX;
    public UInt32 dwY;
    public UInt32 dwXSize;
    public UInt32 dwYSize;
    public UInt32 dwXCountChars;
    public UInt32 dwYCountChars;
    public UInt32 dwFillAttribute;
    public UInt32 dwFlags;
    public UInt16 wShowWindow;
    public UInt16 cbReserved2;
    public IntPtr lpReserved2;
    public IntPtr hStdInput;
    public IntPtr hStdOutput;
    public IntPtr hStdError;
  }

  [StructLayout(LayoutKind.Sequential)]
  private struct PROCESS_INFORMATION {
    public IntPtr hProcess;
    public IntPtr hThread;
    public UInt32 dwProcessId;
    public UInt32 dwThreadId;
  }

  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  private static extern bool CreateProcessW(
    string applicationName,
    StringBuilder commandLine,
    IntPtr processAttributes,
    IntPtr threadAttributes,
    bool inheritHandles,
    UInt32 creationFlags,
    IntPtr environment,
    string currentDirectory,
    ref STARTUPINFO startupInfo,
    out PROCESS_INFORMATION processInformation
  );

  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  private static extern IntPtr CreateJobObject(
    IntPtr jobAttributes,
    string name
  );

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool SetInformationJobObject(
    IntPtr job,
    Int32 informationClass,
    IntPtr information,
    UInt32 informationLength
  );

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool AssignProcessToJobObject(
    IntPtr job,
    IntPtr process
  );

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern UInt32 ResumeThread(IntPtr thread);

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern UInt32 WaitForSingleObject(
    IntPtr handle,
    UInt32 milliseconds
  );

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool GetExitCodeProcess(
    IntPtr process,
    out UInt32 exitCode
  );

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool TerminateProcess(IntPtr process, UInt32 exitCode);

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool TerminateJobObject(IntPtr job, UInt32 exitCode);

  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool CloseHandle(IntPtr handle);

  private static void ThrowLastError(string operation) {
    throw new Win32Exception(
      Marshal.GetLastWin32Error(),
      "Schema-9 supervisor could not " + operation + "."
    );
  }

  public static Int32 Run(
    string commandLine,
    string workingDirectory,
    IntPtr environment
  ) {
    IntPtr job = CreateJobObject(IntPtr.Zero, null);
    if (job == IntPtr.Zero) {
      ThrowLastError("create its containment job");
    }
    PROCESS_INFORMATION process = new PROCESS_INFORMATION();
    bool processCreated = false;
    try {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits =
        new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
      limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      Int32 limitsSize = Marshal.SizeOf(limits);
      IntPtr limitsPointer = Marshal.AllocHGlobal(limitsSize);
      try {
        Marshal.StructureToPtr(limits, limitsPointer, false);
        if (!SetInformationJobObject(
          job,
          JobObjectExtendedLimitInformation,
          limitsPointer,
          (UInt32)limitsSize
        )) {
          ThrowLastError("configure its containment job");
        }
      } finally {
        Marshal.FreeHGlobal(limitsPointer);
      }

      STARTUPINFO startup = new STARTUPINFO();
      startup.cb = (UInt32)Marshal.SizeOf(startup);
      if (!CreateProcessW(
        null,
        new StringBuilder(commandLine),
        IntPtr.Zero,
        IntPtr.Zero,
        true,
        CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
        environment,
        workingDirectory,
        ref startup,
        out process
      )) {
        ThrowLastError("create the bounded command");
      }
      processCreated = true;
      if (!AssignProcessToJobObject(job, process.hProcess)) {
        ThrowLastError("assign the bounded command to its containment job");
      }
      if (ResumeThread(process.hThread) == UInt32.MaxValue) {
        ThrowLastError("resume the bounded command");
      }
      WaitForSingleObject(process.hProcess, INFINITE);
      UInt32 exitCode;
      if (!GetExitCodeProcess(process.hProcess, out exitCode)) {
        ThrowLastError("read the bounded command exit code");
      }
      CloseHandle(process.hThread);
      process.hThread = IntPtr.Zero;
      CloseHandle(process.hProcess);
      process.hProcess = IntPtr.Zero;
      if (!TerminateJobObject(job, 1)) {
        ThrowLastError("terminate descendants after the bounded command exited");
      }
      if (WaitForSingleObject(job, JOB_DRAIN_TIMEOUT_MILLISECONDS) != WAIT_OBJECT_0) {
        throw new TimeoutException(
          "Schema-9 containment job did not drain after termination."
        );
      }
      return unchecked((Int32)exitCode);
    } finally {
      if (processCreated && process.hProcess != IntPtr.Zero) {
        TerminateProcess(process.hProcess, 1);
      }
      if (process.hThread != IntPtr.Zero) {
        CloseHandle(process.hThread);
      }
      if (process.hProcess != IntPtr.Zero) {
        CloseHandle(process.hProcess);
      }
      CloseHandle(job);
    }
  }
}
'@
$commandLine = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String("${encodedCommand}")
)
$workingDirectory = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String("${encodedDirectory}")
)
$encodedEnvironment = [Console]::In.ReadToEnd()
$environmentBytes = [Convert]::FromBase64String($encodedEnvironment)
$environmentPointer = [Runtime.InteropServices.Marshal]::AllocHGlobal(
  $environmentBytes.Length
)
try {
  [Runtime.InteropServices.Marshal]::Copy(
    $environmentBytes,
    0,
    $environmentPointer,
    $environmentBytes.Length
  )
  exit [Schema9JobSupervisor]::Run(
    $commandLine,
    $workingDirectory,
    $environmentPointer
  )
} finally {
  [Runtime.InteropServices.Marshal]::FreeHGlobal($environmentPointer)
}
`;
}

export function preserveSchema9PrimaryWithCleanup(
  primary: Error,
  cleanupFailure: unknown,
  message: string,
): Error {
  if (cleanupFailure === undefined) {
    return primary;
  }
  return new AggregateError([primary, cleanupFailure], message);
}

async function terminateSchema9ProcessTree(child: ChildProcess): Promise<void> {
  const pid = child.pid;
  if (pid === undefined) {
    throw new Error("Schema-9 child process has no authenticated PID.");
  }
  if (process.platform === "win32") {
    const taskkill = await authenticatedWindowsTaskkill();
    await new Promise<void>((accept, reject) => {
      execFile(
        taskkill.path,
        ["/PID", String(pid), "/T", "/F"],
        {
          cwd: dirname(taskkill.path),
          env: taskkill.environment,
          windowsHide: true,
          timeout: TASKKILL_TIMEOUT_MS,
        },
        (error, _stdout, stderr) => {
          void assertAuthenticatedSystemToolUnchanged(taskkill).then(
            () => {
              if (error === null) {
                accept();
                return;
              }
              reject(new Error(
                "Windows could not prove Schema-9 process-tree cleanup.",
                {
                  cause: error instanceof Error
                    ? error
                    : new Error(stderr),
                },
              ));
            },
            reject,
          );
        },
      );
    });
    return;
  }
  try {
    process.kill(-pid, "SIGKILL");
  } catch (error: unknown) {
    if (!isNodeError(error, "ESRCH")) {
      throw error;
    }
  }
}

async function authenticatedWindowsTaskkill(): Promise<AuthenticatedSystemTool> {
  const { root, system32 } = await authenticatedWindowsSystemDirectories();
  const path = await realpath(join(system32, "taskkill.exe"));
  const child = relative(system32, path);
  if (
    child === ""
    || child === ".."
    || child.startsWith(`..${sep}`)
    || isAbsolute(child)
  ) {
    throw new TypeError("Windows taskkill escaped the OS system directory.");
  }
  const metadata = await lstat(path, { bigint: true });
  if (!metadata.isFile()) {
    throw new TypeError("Windows taskkill is not a regular system file.");
  }
  return Object.freeze({
    path,
    dev: metadata.dev,
    ino: metadata.ino,
    size: metadata.size,
    mtimeNs: metadata.mtimeNs,
    ctimeNs: metadata.ctimeNs,
    environment: Object.freeze({
      SystemRoot: root,
      WINDIR: root,
      ComSpec: join(system32, "cmd.exe"),
      PATH: system32,
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
    }),
  });
}

async function authenticatedWindowsPowerShell(): Promise<AuthenticatedSystemTool> {
  const { root, system32 } = await authenticatedWindowsSystemDirectories();
  const powerShellDirectory = await realpath(join(
    system32,
    "WindowsPowerShell",
    "v1.0",
  ));
  const directoryChild = relative(system32, powerShellDirectory);
  if (
    directoryChild === ""
    || directoryChild === ".."
    || directoryChild.startsWith(`..${sep}`)
    || isAbsolute(directoryChild)
  ) {
    throw new TypeError("Windows PowerShell escaped the OS system directory.");
  }
  const path = await realpath(join(powerShellDirectory, "powershell.exe"));
  const executableChild = relative(powerShellDirectory, path);
  if (
    executableChild === ""
    || executableChild === ".."
    || executableChild.startsWith(`..${sep}`)
    || isAbsolute(executableChild)
  ) {
    throw new TypeError("Windows PowerShell resolved outside its system directory.");
  }
  const metadata = await lstat(path, { bigint: true });
  if (!metadata.isFile()) {
    throw new TypeError("Windows PowerShell is not a regular system file.");
  }
  return Object.freeze({
    path,
    dev: metadata.dev,
    ino: metadata.ino,
    size: metadata.size,
    mtimeNs: metadata.mtimeNs,
    ctimeNs: metadata.ctimeNs,
    environment: Object.freeze({
      SystemRoot: root,
      WINDIR: root,
      ComSpec: join(system32, "cmd.exe"),
      PATH: system32,
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
    }),
  });
}

async function authenticatedWindowsSystemDirectories(): Promise<Readonly<{
  root: string;
  system32: string;
}>> {
  if (process.platform !== "win32") {
    throw new TypeError("Windows system directories require Windows.");
  }
  const root = await realpath(WINDOWS_SYSTEM_ROOT_ALIAS);
  const system32 = await realpath(join(WINDOWS_SYSTEM_ROOT_ALIAS, "System32"));
  const child = relative(root, system32);
  if (
    !isAbsolute(root)
    || child.toLocaleLowerCase("en-US") !== "system32"
  ) {
    throw new TypeError("The OS system-directory alias resolved unexpectedly.");
  }
  const rootMetadata = await lstat(root);
  const system32Metadata = await lstat(system32);
  if (!rootMetadata.isDirectory() || !system32Metadata.isDirectory()) {
    throw new TypeError("The OS system-directory alias is not a directory.");
  }
  return Object.freeze({ root, system32 });
}

async function assertAuthenticatedSystemToolUnchanged(
  expected: AuthenticatedSystemTool,
): Promise<void> {
  const actual = await lstat(expected.path, { bigint: true });
  if (
    !actual.isFile()
    || actual.dev !== expected.dev
    || actual.ino !== expected.ino
    || actual.size !== expected.size
    || actual.mtimeNs !== expected.mtimeNs
    || actual.ctimeNs !== expected.ctimeNs
  ) {
    throw new Error("Authenticated Windows cleanup tool changed during use.");
  }
}

function abortOnce(
  controller: AbortController,
  signal: Schema9TerminationSignal,
  exitCode: number,
): void {
  if (!controller.signal.aborted) {
    controller.abort(new Schema9TerminationError(signal, exitCode));
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error("Schema-9 command was interrupted.");
  }
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error
    && "code" in error
    && error.code === code;
}
