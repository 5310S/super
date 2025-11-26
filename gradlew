#!/usr/bin/env sh

##############################################################################
# Minimal Gradle wrapper script.
# Uses the project-local wrapper JAR to bootstrap Gradle without requiring a
# system-wide Gradle installation.
##############################################################################

set -e

APP_HOME="$(cd "$(dirname "$0")" && pwd)"
CLASSPATH="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"
MAIN_CLASS="org.gradle.wrapper.GradleWrapperMain"

if [ -n "$JAVA_HOME" ]; then
  JAVA_BIN="$JAVA_HOME/bin/java"
else
  JAVA_BIN="java"
fi

exec "$JAVA_BIN" -classpath "$CLASSPATH" "$MAIN_CLASS" "$@"
