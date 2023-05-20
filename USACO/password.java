import java.util.*;
import java.io.*;
import java.lang.*;

public class password
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("password.in"));
      String s = in.next();
      long n = in.nextLong() - 1;
      long l = s.length();
      in.close();
      
      while(l <= n)
         l = l * 2;
      while(l > s.length())
      {
         if(n == l / 2)
            n--;
         else if(n > l / 2)
            n = n - (l / 2) - 1;
         else
            l /= 2;
      }
      
      System.out.println(s.charAt((int)n));
   }
}