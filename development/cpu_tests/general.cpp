
#include "gtest/gtest.h"
#include "wbCore.h"

using namespace wb;
using namespace wb::io;

/// <summary>
/// Sets the file's timestamp(s).  Any timestamp specified as nullptr will not be updated.
/// </summary>			
inline void SetFileTime(FileStream* p_fs, DateTime& creation_time, DateTime& last_access_time, DateTime& last_write_time)
{
    FILETIME ft_creation = creation_time.ToFILETIME();
    FILETIME ft_last_access = last_access_time.ToFILETIME();
    FILETIME ft_last_write = last_write_time.ToFILETIME();
    if (!::SetFileTime(p_fs->GetHandle(), &ft_creation, &ft_last_access, &ft_last_write))
        Exception::ThrowFromWin32(::GetLastError());
}

TEST(General, DateTime_FILETIME)
{
    auto dt = DateTime(2024, 5, 21, 14, 20, 30, 1, 0);
    FILETIME ft = dt.ToFILETIME();
    auto dt2 = DateTime(ft);
    if (dt.asUTC().GetHour() != dt2.asUTC().GetHour())
        throw Exception("Self-test has failed (FILETIME conversion): " + dt.ToString() + " does not match " + dt2.ToString());

    {
        auto fw = wb::io::StreamWriter(wb::memory::r_ptr<Stream>::responsible(
            new FileStream("self_test.txt", FileMode::Create)
        ));
        fw.WriteLine("Testing");

        SetFileTime((FileStream*)fw.GetStream().get(), dt, dt, dt);
        fw.GetStream().get()->Flush();
    }
    auto fi = wb::io::FileInfo(L"self_test.txt");
    DateTime dt3 = fi.GetLastWriteTime();
    dt = dt.asUTC();
    dt3 = dt3.asUTC();
    if (dt.GetDay() != dt3.GetDay() || dt.GetHour() != dt3.GetHour() || dt.GetMinute() != dt3.GetMinute())
        throw Exception("Self-test has failed (FILETIME save): " + dt.ToString() + " does not match " + dt3.ToString());
    //wb::io::File::Delete(L"self_test.txt");    

    std::cout << ("Wrote self_test.txt with target DateTime (UTC) of: " + dt.asUTC().asISO8601());
    std::cout << ("Readback self_test.txt with DateTime (UTC) of:     " + dt3.asUTC().asISO8601());
    std::cout << ("The FILETIME of the target DateTime was:           low=" + wb::to_hex_string(ft.dwLowDateTime) + "   high=" + wb::to_hex_string(ft.dwHighDateTime));
}

