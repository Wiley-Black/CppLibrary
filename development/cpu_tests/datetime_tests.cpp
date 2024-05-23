
#include "gtest/gtest.h"
#include "wbCore.h"

using namespace wb;
using namespace wb::io;

string st_to_string(SYSTEMTIME st)
{
    return std::to_string((int)st.wYear) + "-" + std::to_string((int)st.wMonth) + "-" + std::to_string((int)st.wDay) + " " + std::to_string((int)st.wHour) + ":" + std::to_string((int)st.wMinute) + ":" + std::to_string((int)st.wSecond);
}

/// <summary>
/// Sets the file's timestamp(s).  Any timestamp specified as nullptr will not be updated.
/// </summary>			
inline void SetFileTime(FileStream* p_fs, DateTime& creation_time, DateTime& last_access_time, DateTime& last_write_time)
{   
    #if 1
    FILETIME ft_creation = creation_time.ToFILETIME();
    FILETIME ft_last_access = last_access_time.ToFILETIME();
    FILETIME ft_last_write = last_write_time.ToFILETIME();
    #else    
    SYSTEMTIME st_creation = creation_time.asUTC().ToSYSTEMTIME();
    SYSTEMTIME st_last_access = last_access_time.asUTC().ToSYSTEMTIME();
    SYSTEMTIME st_last_write = last_write_time.asUTC().ToSYSTEMTIME();
    FILETIME ft_creation, ft_last_access, ft_last_write;    
    if (!SystemTimeToFileTime(&st_creation, &ft_creation)) Exception::ThrowFromWin32(::GetLastError());    
    if (!SystemTimeToFileTime(&st_last_access, &ft_last_access)) Exception::ThrowFromWin32(::GetLastError());
    if (!SystemTimeToFileTime(&st_last_write, &ft_last_write)) Exception::ThrowFromWin32(::GetLastError());
    #endif

    if (!::SetFileTime(p_fs->GetHandle(), &ft_creation, &ft_last_access, &ft_last_write))
        Exception::ThrowFromWin32(::GetLastError());
}

SYSTEMTIME GetFileLastWriteTime(const TCHAR* pszFilename, bool as_local_time)
{
    HANDLE hFile = CreateFile(pszFilename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        throw IOException("Unable to open file to verify file time: " + wb::to_string(pszFilename));
    FILETIME ftCreate, ftAccess, ftWrite;
    if (!::GetFileTime(hFile, &ftCreate, &ftAccess, &ftWrite))
        throw IOException("Unable to retrieve file time: " + wb::to_string(pszFilename));
    
    // Convert the last-write time to local time.
    SYSTEMTIME stUTC, stLocal;
    FileTimeToSystemTime(&ftWrite, &stUTC);
    if (!as_local_time)
        return stUTC;
    SystemTimeToTzSpecificLocalTime(NULL, &stUTC, &stLocal);
    return stLocal;
}

TEST(DateTime, LocalTime)
{
    auto dt_utc = DateTime(2024, 5, 21, 15, 20, 30, 1000000, 0);                     // Z
    auto dt_est = DateTime(2024, 5, 21, 10, 20, 30, 1000000, -(5 * 60));             // -05:00 offset
    auto dt_odd = DateTime(2024, 5, 21, 11, 00, 30, 1000000, -(4 * 60 + 20));        // -04:20 offset
    auto dt_CEST = DateTime(2024, 5, 21, 17, 20, 30, 1000000, (2 * 60));             // GMT+2

    const string utc = "2024-05-21T15:20:30.001000Z";
    ASSERT_TRUE(IsEqual(dt_utc.asISO8601(), utc)) << "\nActual: " << dt_utc.asISO8601() << "\nExpected:" << utc;
    string expected = "2024-05-21T10:20:30.001000-05:00";
    ASSERT_TRUE(IsEqual(dt_est.asISO8601(), expected)) << "\nActual: " << dt_est.asISO8601() << "\nExpected:" << expected;
    expected = "2024-05-21T11:00:30.001000-04:20";
    ASSERT_TRUE(IsEqual(dt_odd.asISO8601(), expected)) << "\nActual: " << dt_odd.asISO8601() << "\nExpected:" << expected;
    expected = "2024-05-21T17:20:30.001000+02:00";
    ASSERT_TRUE(IsEqual(dt_CEST.asISO8601(), expected)) << "\nActual: " << dt_CEST.asISO8601() << "\nExpected:" << expected;

    string actual = dt_est.asUTC().asISO8601();
    ASSERT_TRUE(IsEqual(actual, utc)) << "\nActual: " << actual << "\nExpected: " << utc;
    actual = dt_odd.asUTC().asISO8601();
    ASSERT_TRUE(IsEqual(actual, utc)) << "\nActual: " << actual << "\nExpected: " << utc;
    actual = dt_CEST.asUTC().asISO8601();
    ASSERT_TRUE(IsEqual(actual, utc)) << "\nActual: " << actual << "\nExpected: " << utc;

    ASSERT_TRUE(dt_est.asUTC() == dt_utc);
    ASSERT_TRUE(dt_odd.asUTC() == dt_utc);
    ASSERT_TRUE(dt_CEST.asUTC() == dt_utc);

    // TODO: would be nice to test asLocal(), but it changes with time and location.  The file test below will
    // give it a test though.
    // std::cout << "As local time: " << dt_utc.asLocalTime().asISO8601() << std::endl;
}

TEST(DateTime, FILETIME)
{
    Path file_path("file_timestamp.txt");

    auto dt = DateTime(2024, 5, 21, 14, 20, 30, 1, 0);

    std::cout << "utc bias = " << (int)(dt.GetBias() / 3600) << " hr\n";
    std::cout << "local bias = " << (int)(dt.asLocalTime().GetBias() / 3600) << " hr\n";

    // If UTC is:           2024-05-21 14:30:01Z
    // EST (UTC-05) is:     2024-05-21 10:30:01-05:00
    // dt should be specified in UTC time.  Assume ET has a bias of +4 

    FILETIME ft = dt.ToFILETIME();
    auto dt2 = DateTime(ft);
    if (dt.asUTC().GetHour() != dt2.asUTC().GetHour())
        FAIL() << "FILETIME conversion has failed: " << dt.ToString() << " does not match " << dt2.ToString();

    if (dt.GetHour() != 14)
        FAIL() << "DateTime hour (" << dt.GetHour() << ") expected to be provided value (14), but wasn't.";

    std::cout << ("Target time [UTC]:             " + dt.asUTC().asISO8601()) << std::endl;
    std::cout << ("Target time [Local Time]:      " + dt.asLocalTime().asISO8601()) << std::endl;

    {        
        SYSTEMTIME stUTC;
        FileTimeToSystemTime(&ft, &stUTC);
        if (dt.GetDay() != stUTC.wDay || dt.GetHour() != stUTC.wHour || dt.GetMinute() != stUTC.wMinute)
            FAIL() << "Conversion timestamp write and SYSTEMTIME mismatch [UTC]: " << dt.ToString() << " does not match " << st_to_string(stUTC) << ".";
    }    

    {
        auto fw = wb::io::StreamWriter(wb::memory::r_ptr<Stream>::responsible(
            new FileStream(file_path, FileMode::Create)
        ));
        fw.WriteLine("Testing");

        SetFileTime((FileStream*)fw.GetStream().get(), dt, dt, dt);
        fw.GetStream().get()->Flush();
    }
    auto fi = wb::io::FileInfo(file_path);
    DateTime dt3 = fi.GetLastWriteTime();
    dt = dt.asUTC();
    dt3 = dt3.asUTC();
    if (dt.GetDay() != dt3.GetDay() || dt.GetHour() != dt3.GetHour() || dt.GetMinute() != dt3.GetMinute())
        FAIL() << "File timestamp write and FileInfo retrieval has failed: " << dt.ToString() << " does not match " << dt3.ToString();

    auto stUTC = GetFileLastWriteTime(file_path.to_osstring().c_str(), false);
    if (dt.GetDay() != stUTC.wDay || dt.GetHour() != stUTC.wHour || dt.GetMinute() != stUTC.wMinute)
        FAIL() << "File timestamp write and SYSTEMTIME mismatch [UTC]: " << dt.ToString() << " does not match retrieved time " << st_to_string(stUTC) << ".";

    auto dtLocal = dt.asLocalTime();
    auto stLocal = GetFileLastWriteTime(file_path.to_osstring().c_str(), true);
    if (dtLocal.GetDay() != stLocal.wDay || dtLocal.GetHour() != stLocal.wHour || dtLocal.GetMinute() != stLocal.wMinute)
        FAIL() << "File timestamp write and SYSTEMTIME mismatch [local time]: " << dt.ToString() << " does not match retrieved time " << st_to_string(stLocal) << ".";
    
    std::cout << ("Wrote '" + file_path.to_string() + "' with target DateTime(UTC) of:     " + dt.asUTC().asISO8601()) << std::endl;
    std::cout << ("Readback with DateTime (UTC) of:             " + dt3.asUTC().asISO8601()) << std::endl;
    std::cout << ("Readback with DateTime (Local Time) of:      " + dt3.asLocalTime().asISO8601()) << std::endl;
    std::cout << ("The FILETIME of the target DateTime was:     low=" + wb::to_hex_string(ft.dwLowDateTime) + "   high=" + wb::to_hex_string(ft.dwHighDateTime)) << std::endl;
}

